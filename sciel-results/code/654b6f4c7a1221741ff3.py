# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    Generates a path for the moving sofa problem using a circular arc model.
    The inner corner of the L-shaped corridor moves along a circular arc
    centered at (-1, -1) with radius sqrt(2), from (0, 0) to (-2, 0).
    This path ensures that the rotation of the corner relative to the center
    is exactly 90 degrees, matching the required rotation of the sofa.
    """
    # Parameters for the circular arc
    center_x = -1.0
    center_y = -1.0
    radius = np.sqrt(2.0)
    
    # Start and end angles in radians
    # Start point (0,0) relative to center (-1,-1) is (1,1), angle 45 deg
    # End point (-2,0) relative to center (-1,-1) is (-1,1), angle 135 deg
    start_angle = np.pi / 4.0
    end_angle = 3.0 * np.pi / 4.0
    
    # Number of intermediate points to generate
    # We exclude the start and end points as they are added by the evaluator.
    # Using 20 points provides a good approximation of the arc.
    n_points = 20
    
    # Generate angles strictly between start and end
    angles = np.linspace(start_angle, end_angle, n_points + 2)[1:-1]
    
    # Calculate positions
    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)
    
    # Calculate rotation angles in degrees
    # a = (phi - start_angle) in degrees
    # Since start_angle corresponds to a=0 and end_angle to a=90
    a_coords = (angles - start_angle) * (180.0 / np.pi)
    
    # Construct the path array
    path = np.column_stack((x_coords, y_coords, a_coords))
    
    return path


# EVOLVE-BLOCK-END
