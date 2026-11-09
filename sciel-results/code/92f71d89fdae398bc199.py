import numpy as np


def moving_sofa_3d_path() -> np.ndarray:
    """
    Generate a parametric path for the 3D moving sofa problem.
    
    The path is constructed analytically to navigate the L-shaped 3D corridor
    with two 90-degree turns. The strategy uses early rotation to minimize
    the bounding box during corner transitions.
    """
    keyframes = []
    
    # Phase 1: Move from x=-3 to x=-0.5 in region 1 (x<=0, 0<=y<=1, 0<=z<=1)
    # Keep y=0.5, z=0.5 centered, no rotation
    for i in range(20):
        t = i / 19.0
        x = -3.0 + 2.5 * t  # from -3 to -0.5
        y = 0.5
        z = 0.5
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        keyframes.append([x, y, z, yaw, pitch, roll])
    
    # Phase 2: Transition through first corner (x=0, y=0)
    # Rotate yaw from 0 to 90 degrees while moving from x=-0.5,y=0.5 to x=0.5,y=0.5
    # This aligns the sofa with the y-direction for region 2
    for i in range(30):
        t = i / 29.0
        # Smoothstep for smoother rotation
        s = t * t * (3 - 2 * t)
        x = -0.5 + 1.0 * s  # from -0.5 to 0.5
        y = 0.5
        z = 0.5
        yaw = 90.0 * s
        pitch = 0.0
        roll = 0.0
        keyframes.append([x, y, z, yaw, pitch, roll])
    
    # Phase 3: Move along y from 0.5 to 5.0 in region 2 (0<=x<=1, 0<=y<=5, 0<=z<=1)
    # Keep x=0.5 centered, z=0.5 centered, yaw=90
    for i in range(20):
        t = i / 19.0
        x = 0.5
        y = 0.5 + 4.5 * t  # from 0.5 to 5.0
        z = 0.5
        yaw = 90.0
        pitch = 0.0
        roll = 0.0
        keyframes.append([x, y, z, yaw, pitch, roll])
    
    # Phase 4: Transition through second corner (y=5, z=0)
    # Rotate pitch from 0 to 90 degrees while moving from z=0.5 to z=0.5
    # This aligns the sofa with the z-direction for region 3
    for i in range(30):
        t = i / 29.0
        s = t * t * (3 - 2 * t)
        x = 0.5
        y = 5.0
        z = 0.5
        yaw = 90.0
        pitch = 90.0 * s
        roll = 0.0
        keyframes.append([x, y, z, yaw, pitch, roll])
    
    # Phase 5: Move along z from 0.5 to 7.0 in region 3 (0<=x<=1, 5<=y<=6, z>=0)
    # Keep x=0.5 centered, y=5.5 centered, pitch=90
    for i in range(20):
        t = i / 19.0
        x = 0.5
        y = 5.5
        z = 0.5 + 6.5 * t  # from 0.5 to 7.0
        yaw = 90.0
        pitch = 90.0
        roll = 0.0
        keyframes.append([x, y, z, yaw, pitch, roll])
    
    # Phase 6: Final rotation to match end pose (roll=90)
    for i in range(10):
        t = i / 9.0
        s = t * t * (3 - 2 * t)
        x = 0.0
        y = 6.0
        z = 7.0
        yaw = 90.0
        pitch = 90.0
        roll = 90.0 * s
        keyframes.append([x, y, z, yaw, pitch, roll])
    
    path = np.array(keyframes, dtype=np.float64)
    return path
