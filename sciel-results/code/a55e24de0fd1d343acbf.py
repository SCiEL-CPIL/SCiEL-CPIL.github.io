# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    sqrt3 = np.sqrt(3)
    
    # Place 12 unit hexagons in a 4x3 staggered grid (pointy-top, angle=0)
    centers = []
    for row in range(3):
        y = row * 1.5
        x_offset = (sqrt3 / 2) * (row % 2)
        for col in range(4):
            x = x_offset + col * sqrt3
            centers.append([x, y])
    
    centers = np.array(centers)
    inner_hex_data = np.column_stack([centers, np.zeros(12)])
    
    # Compute all vertices of all inner hexagons (each has 6 vertices at distance 1 from center)
    all_vertices = []
    for i in range(12):
        cx, cy = centers[i]
        for j in range(6):
            angle = np.deg2rad(60 * j)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    def enclosing_size(angle_deg):
        a = np.deg2rad(angle_deg)
        normals = np.array([[np.cos(a + np.deg2rad(30 + 60*i)), np.sin(a + np.deg2rad(30 + 60*i))] for i in range(6)])
        
        c = np.mean(all_vertices, axis=0)
        
        max_proj = 0.0
        for v in all_vertices:
            for n in normals:
                proj = np.dot(n, v - c)
                if proj > max_proj:
                    max_proj = proj
        
        s = max_proj / (np.sqrt(3) / 2)
        return s
    
    result = minimize_scalar(enclosing_size, bounds=(0, 60), method='bounded')
    best_angle = result.x
    
    # Recompute with optimal angle
    a = np.deg2rad(best_angle)
    normals = np.array([[np.cos(a + np.deg2rad(30 + 60*i)), np.sin(a + np.deg2rad(30 + 60*i))] for i in range(6)])
    
    c = np.mean(all_vertices, axis=0)
    
    max_proj = 0.0
    for v in all_vertices:
        for n in normals:
            proj = np.dot(n, v - c)
            if proj > max_proj:
                max_proj = proj
    
    s = max_proj / (np.sqrt(3) / 2)
    
    outer_hex_data = np.array([c[0], c[1], best_angle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
