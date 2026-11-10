import numpy as np

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Centers for the 11 hexagons
    # 1 center, 6 in a ring at distance 2, 4 in the diagonal gaps at distance 2*sqrt(3)
    centers = [
        [0, 0],
        [2, 0],
        [1, np.sqrt(3)],
        [-1, np.sqrt(3)],
        [-2, 0],
        [-1, -np.sqrt(3)],
        [1, -np.sqrt(3)],
        [3, np.sqrt(3)],
        [3, -np.sqrt(3)],
        [-3, np.sqrt(3)],
        [-3, -np.sqrt(3)]
    ]
    
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0  # Angle 0 degrees
        
    outer_hex_data = np.array([0, 0, 30])  # Centered at origin, angle 30 degrees to align vertices with the 30, 90, ... directions of the cluster
    outer_hex_side_length = 2 * np.sqrt(3) + 1
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
