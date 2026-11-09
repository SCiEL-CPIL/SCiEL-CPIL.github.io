import numpy as np

def moving_sofa_3d_path() -> np.ndarray:
    # Key poses
    # Start is prepended: (-3, 0, 0, 0, 0, 0)
    # End is appended: (0, 6, 7, 90, 90, 90)
    
    path = [
        # Move to center of first leg
        [-0.5, 0.5, 0.5, 0, 0, 0],
        
        # Approach corner 1
        [0.0, 0.5, 0.5, 0, 0, 0],
        
        # Rotate in corner 1 (Center 0.5, 0.5, 0.5)
        # Use smaller steps to ensure interpolation is smooth and checks pass
        [0.5, 0.5, 0.5, 22.5, 0, 0],
        [0.5, 0.5, 0.5, 45.0, 0, 0],
        [0.5, 0.5, 0.5, 67.5, 0, 0],
        [0.5, 0.5, 0.5, 90.0, 0, 0],
        
        # Move along second leg
        [0.5, 1.0, 0.5, 90, 0, 0],
        [0.5, 2.0, 0.5, 90, 0, 0],
        [0.5, 3.0, 0.5, 90, 0, 0],
        [0.5, 4.0, 0.5, 90, 0, 0],
        [0.5, 5.0, 0.5, 90, 0, 0],
        
        # Approach corner 2
        [0.5, 5.5, 0.5, 90, 0, 0],
        
        # Rotate in corner 2 (Center 0.5, 5.5, 0.5)
        # Decompose rotation into single-axis steps
        # Step 1: Pitch 0 -> 90
        [0.5, 5.5, 0.5, 90, 22.5, 0],
        [0.5, 5.5, 0.5, 90, 45.0, 0],
        [0.5, 5.5, 0.5, 90, 67.5, 0],
        [0.5, 5.5, 0.5, 90, 90.0, 0],
        
        # Step 2: Roll 0 -> 90
        [0.5, 5.5, 0.5, 90, 90, 22.5],
        [0.5, 5.5, 0.5, 90, 90, 45.0],
        [0.5, 5.5, 0.5, 90, 90, 67.5],
        [0.5, 5.5, 0.5, 90, 90, 90.0],
        
        # Move along third leg
        [0.5, 5.5, 1.0, 90, 90, 90],
        [0.5, 5.5, 2.0, 90, 90, 90],
        [0.5, 5.5, 3.0, 90, 90, 90],
        [0.5, 5.5, 4.0, 90, 90, 90],
        [0.5, 5.5, 5.0, 90, 90, 90],
        [0.5, 5.5, 6.0, 90, 90, 90],
        
        # Approach end
        [0.0, 6.0, 7.0, 90, 90, 90]
    ]
    
    return np.array(path, dtype=np.float64)
