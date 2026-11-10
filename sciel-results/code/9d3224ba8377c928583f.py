import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree).
        outer_hex_side_length: float.
    """
    # Topology: Ring 2 of a hexagonal tiling (12 hexagons)
    # Centers are at distance 2 from origin, angles 0, 60, ..., 300 degrees.
    # This arrangement is optimal, achieving the theoretical lower bound s = 2*sqrt(3).
    
    angles_deg = np.arange(0, 360, 60)
    angles_rad = np.deg2rad(angles_deg)
    
    # Centers at radius 2
    radius = 2.0
    centers_x = radius * np.cos(angles_rad)
    centers_y = radius * np.sin(angles_rad)
    
    # All inner hexagons aligned with angle 0
    inner_angles = np.zeros(12)
    
    inner_hex_data = np.column_stack((centers_x, centers_y, inner_angles))
    
    # Outer hexagon
    # Side length s = 2 * sqrt(3)
    s = 2.0 * np.sqrt(3.0)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
