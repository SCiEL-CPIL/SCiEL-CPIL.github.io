# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path for the moving sofa problem.
    The path is defined by the position (x, y) of the inner corner of the L-shaped corridor
    and its rotation angle a in degrees.
    
    The optimal path keeps the center of the "corner square" of the corridor moving
    linearly from (0.5, 0.5) to (-2.5, 0.5) while rotating from 0 to 90 degrees.
    This corresponds to the inner corner moving from (0, 0) to (-2, 0) with y=0.
    """
    # Generate angles from 0 to 90 degrees in 5-degree steps
    angles = np.arange(0, 91, 5.0)
    
    # Calculate x and y positions
    # x(a) = -2 * (a / 90)
    # y(a) = 0
    x = -2.0 * angles / 90.0
    y = np.zeros_like(angles)
    
    # Stack into (n, 3) array
    path = np.column_stack((x, y, angles))
    
    return path

# EVOLVE-BLOCK-END
