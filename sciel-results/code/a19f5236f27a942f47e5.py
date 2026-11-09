import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    # Side length of outer hexagon
    s = 4.0
    
    # Inner hexagon centers
    # 1 central
    centers = []
    centers.append([0.0, 0.0, 0.0])
    
    # 6 surrounding (flower)
    r_flower = np.sqrt(3)
    for i in range(6):
        angle = np.deg2rad(30 + 60 * i)
        x = r_flower * np.cos(angle)
        y = r_flower * np.sin(angle)
        centers.append([x, y, 0.0])
        
    # 5 at vertices (angles 0, 60, 120, 180, 240)
    r_vertex = s - 1.0
    for i in range(5):
        angle = np.deg2rad(60 * i)
        x = r_vertex * np.cos(angle)
        y = r_vertex * np.sin(angle)
        centers.append([x, y, 0.0])
        
    inner_hex_data = np.array(centers)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
