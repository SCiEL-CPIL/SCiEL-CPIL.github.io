# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses an analytic construction with a small separation gap to ensure validity.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin

    # Parameters
    delta = 1e-5 # Separation gap between hexagons
    d = np.sqrt(3) + delta # Distance between centers of adjacent hexagons
    
    # Inner Hexagon Centers
    # 1 Center
    c0 = np.array([0.0, 0.0])
    
    # 6 Neighbors
    c1 = np.array([d, 0.0])
    c2 = np.array([d * 0.5, d * np.sqrt(3) * 0.5])
    c3 = np.array([-d * 0.5, d * np.sqrt(3) * 0.5])
    c4 = np.array([-d, 0.0])
    c5 = np.array([-d * 0.5, -d * np.sqrt(3) * 0.5])
    c6 = np.array([d * 0.5, -d * np.sqrt(3) * 0.5])
    
    # 4 Outer Extensions
    c7 = np.array([2 * d, 0.0])
    c8 = np.array([d, d * np.sqrt(3)])
    c9 = np.array([-d, d * np.sqrt(3)])
    c10 = np.array([-2 * d, 0.0])
    
    centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    
    # All inner hexagons have angle 0
    angles = np.zeros(11)
    
    inner_hex_data = np.column_stack((centers[:, 0], centers[:, 1], angles))
    
    # Calculate required outer apothem
    # Max projection is d*sqrt(3) + sqrt(3)/2
    A_req = d * np.sqrt(3) + np.sqrt(3) / 2.0
    
    # Add margin
    margin = 1e-4
    A_out = A_req + margin
    
    # Outer hexagon side length
    S = A_out * 2.0 / np.sqrt(3.0)
    
    # Outer hexagon centered at origin, angle 0
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, S

# EVOLVE-BLOCK-END
