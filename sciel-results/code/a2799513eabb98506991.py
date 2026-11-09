# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path for moving a sofa through an L-shaped corridor.
    Returns path as (n, 3) array of (x, y, angle_degrees).
    """
    # Angles for intermediate positions
    angles = np.array([10.0, 20.0, 35.0, 50.0, 65.0, 75.0, 82.0, 88.0])
    
    # Position as function of angle
    # The inner corner should move from (0,0) to (-2,0) as angle goes 0 to 90
    # Use a smooth curve
    a_rad = np.radians(angles)
    t = angles / 90.0
    
    # Try a path where x moves linearly and y has a smooth bump
    x = -2.0 * t
    y = 0.4 * np.sin(np.pi * t)
    
    path = np.column_stack([x, y, angles])
    return path


# EVOLVE-BLOCK-END
