import numpy as np
import time

N = 13

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 13 points in a regular 13-gon inscribed in a unit circle.
    This configuration maximizes the minimum triangle area relative to the hull area
    by evenly distributing the points on the boundary of the convex region.
    """
    # Calculate angles for a regular n-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Coordinates: (cos(theta), sin(theta))
    # This creates a regular polygon inscribed in the unit circle
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return points
# EVOLVE-BLOCK-END
