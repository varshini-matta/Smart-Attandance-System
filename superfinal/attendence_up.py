import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

def writefun():
    current_date = datetime.now().strftime("%Y-%m-%d")
    csv_filename = current_date + '.csv'

    # Create the file if it doesn't exist
    with open(csv_filename, 'a', newline='') as file:
        pass

    return csv_filename

def check_name_exists(name, filename):
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == name:
                return True
    return False

def add_name_to_csv(name, filename):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([name, timestamp])
    print(f"Added {name} to {filename}.")

# Load images and encode known faces
known_faces_dir = "Images"
known_faces = []
known_names = []
for name in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, name)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(name)[0])

# Initialize variables
attendance = {}
frame_thickness = 3
font_thickness = 2
font_scale = 0.75

# Initialize video capture
cap = cv2.VideoCapture(0)

filename = writefun()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster face recognition (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR image to RGB for face recognition
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for idx, face_encoding in enumerate(face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Check for a match
        if True in matches:
            matched_indices = [i for i, match in enumerate(matches) if match]
            counts = {}
            for i in matched_indices:
                counts[known_names[i]] = counts.get(known_names[i], 0) + 1
            name = max(counts, key=counts.get)

            # Draw bounding box and label on the frame
            top, right, bottom, left = face_locations[idx]
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), frame_thickness)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            # Update attendance dictionary
            if name not in attendance:
                attendance[name] = True
                print(f"Images/{name}.jpg")

                # Check if the image file exists before reading
                img_path = f"Images/{name}.jpg"
                if os.path.exists(img_path):
                    imgbg = cv2.imread(img_path)
                    imS = cv2.resize(imgbg, (250, 250))
                    cv2.imshow('Attendance System', imS)

                    # Check if attendance has already been taken for this person
                    if not check_name_exists(name, filename):
                        add_name_to_csv(name, filename)
                    else:
                        print("Attendance already taken")

    # Display the resulting frame
    cv2.imshow("Camera with Background", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
