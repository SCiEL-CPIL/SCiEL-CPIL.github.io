<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 12
    # Simple grid arrangement of inner hexagons
    inner_hex_data = np.array(
        [
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ]
    )

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    deadline = time.time() + 600
    
    # Hexagonal grid arrangement for 12 hexagons using axial coordinates
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0)
    ]
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
        return x, y
    
    centers = np.array([axial_to_cartesian(q, r) for q, r in axial_coords])
    angles = np.zeros(12)
    
    def get_hex_vertices(center, side, angle_deg):
        cx, cy = center
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + np.radians(60 * i)
            vx = cx + side * np.cos(theta)
            vy = cy + side * np.sin(theta)
            vertices.append([vx, vy])
        return np.array(vertices)
    
    def compute_enclosing_hex_side(centers, angles, outer_angle):
        all_vertices = []
        for i in range(len(centers)):
            verts = get_hex_vertices(centers[i], 1.0, angles[i])
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        outer_angle_rad = np.radians(outer_angle)
        hex_center = np.mean(all_vertices, axis=0)
        
        max_projection = 0
        for i in range(6):
            normal_angle = outer_angle_rad + np.radians(60 * i + 30)
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            projections = np.dot(all_vertices - hex_center, normal)
            max_projection = max(max_projection, np.max(projections))
        
        s = max_projection / (np.sqrt(3) / 2)
        return s, hex_center
    
    def objective(params):
        if time.time() > deadline - 30:
            return 1e10
        cluster_angle, outer_angle = params
        cluster_angle_rad = np.radians(cluster_angle)
        
        cos_a = np.cos(cluster_angle_rad)
        sin_a = np.sin(cluster_angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_centers = centers @ rotation_matrix.T
        
        s, _ = compute_enclosing_hex_side(rotated_centers, angles, outer_angle)
        return s
    
    best_s = np.inf
    best_params = np.array([0.0, 0.0])
    
    starts = [np.array([0, 0]), np.array([30, 0]), np.array([0, 30]), 
              np.array([15, 15]), np.array([45, 0]), np.array([0, 45]),
              np.array([60, 0]), np.array([10, 20])]
    
    for start in starts:
        if time.time() > deadline - 60:
            break
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                            options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-3})
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    cluster_angle, outer_angle = best_params
    cluster_angle_rad = np.radians(cluster_angle)
    
    cos_a = np.cos(cluster_angle_rad)
    sin_a = np.sin(cluster_angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_centers = centers @ rotation_matrix.T
    
    s, hex_center = compute_enclosing_hex_side(rotated_centers, angles, outer_angle)
    
    inner_hex_data = np.column_stack([rotated_centers, angles])
    outer_hex_data = np.array([hex_center[0], hex_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
>>>>>>> REPLACE
