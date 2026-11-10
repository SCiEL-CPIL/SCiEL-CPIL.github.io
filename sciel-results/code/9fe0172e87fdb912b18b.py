import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    d = np.sqrt(3)  # base distance between adjacent hexagon centers
    
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons at distance d from center
    for i in range(6):
        angle = i * 60.0
        rad = np.radians(angle)
        x = d * np.cos(rad)
        y = d * np.sin(rad)
        inner.append([x, y, 0.0])
    
    # Ring 2: 4 hexagons at distance 2*d, angles 30, 150, 210, 330
    for angle_deg in [30.0, 150.0, 210.0, 330.0]:
        rad = np.radians(angle_deg)
        x = 2 * d * np.cos(rad)
        y = 2 * d * np.sin(rad)
        inner.append([x, y, 0.0])
    
    inner_hex_data = np.array(inner)
    
    # Compute minimum outer hexagon side to contain all inner hexagons
    # Outer hex: pointy-top, centered at origin, side S
    # Edge normals at angles 0, 60, 120, 180, 240, 300
    # Inradius = S * sqrt(3)/2
    # Point (px,py) inside iff max_i |px*cos(a_i) + py*sin(a_i)| <= S*sqrt(3)/2
    
    normals = []
    for i in range(6):
        angle = i * 60.0
        rad = np.radians(angle)
        normals.append([np.cos(rad), np.sin(rad)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for (cx, cy, ang) in inner_hex_data:
        for i in range(6):
            vert_ang = np.radians(ang + i * 60.0)
            vx = cx + np.cos(vert_ang)
            vy = cy + np.sin(vert_ang)
            proj = np.abs(normals[:, 0] * vx + normals[:, 1] * vy)
            max_proj = max(max_proj, float(np.max(proj)))
    
    outer_hex_side_length = 2.0 * max_proj / np.sqrt(3)
    
    # Add tiny margin for floating point safety
    outer_hex_side_length += 1e-9
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(outer_hex_side_length)
