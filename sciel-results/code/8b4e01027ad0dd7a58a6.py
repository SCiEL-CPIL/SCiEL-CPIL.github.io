# EVOLVE-BLOCK-START
import numpy as np
import time


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    # Optimal packing: central hex + 6 in first ring (dist 2) + 5 in second ring (dist 4)
    # Outer hexagon side = 5, rotated 30 degrees
    #
    # First ring at distance 2, angles 0, 60, 120, 180, 240, 300
    # Second ring at distance 4, angles 30, 90, 150, 210, 270 (leaving 330 gap)
    # This leaves a symmetric gap at angle 330 (-30)
    
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    # First ring: distance 2, angles 0, 60, 120, 180, 240, 300
    for i in range(6):
        angle = np.deg2rad(60.0 * i)
        inner.append([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
    # Second ring: distance 4, angles 30, 90, 150, 210, 270
    for i in range(5):
        angle = np.deg2rad(30.0 + 60.0 * i)
        inner.append([4.0 * np.cos(angle), 4.0 * np.sin(angle), 0.0])
    
    inner_hex_data = np.array(inner, dtype=float)
    outer_hex_data = np.array([0.0, 0.0, 30.0])  # rotated 30 degrees
    outer_hex_side_length = 5.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
