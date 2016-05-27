import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face and eye cascade
    face_cascade = cv2.CascadeClassifier('../resources/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../resources/haarcascade_eye.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Show detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Show detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('OpenCV Basic Project', frame)

    # Close program by using key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
