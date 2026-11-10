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
    # We need exactly 12. 
    # Center (0,0)
    # Ring 1 (6): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (6 selected): (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0) -> This makes 13.
    # Let's pick 5 from Ring 2 to make 12? Or a different shape.
    # A 2x3 parallelogram of hexagons?
    # Axial coords for a 2x3 block (q=0..1, r=0..2 with skew)
    # Let's just use the 7 center + 5 from the next ring to form a "hexagon of side 2 with one corner cut off" or similar.
    # Actually, a regular hexagon of side 2 has 7 hexes.
    # To get 12, we can add 5 more.
    # Let's use these 12:
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
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
    
    def compute_enclosing_hex_side(centers, angles, outer_center, outer_angle):
        all_vertices = []
        for i in range(len(centers)):
            verts = get_hex_vertices(centers[i], 1.0, angles[i])
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        outer_angle_rad = np.radians(outer_angle)
        ocx, ocy = outer_center
        
        max_projection = 0
        for i in range(6):
            normal_angle = outer_angle_rad + np.radians(60 * i + 30)
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            # Project vertices relative to outer center
            projections = np.dot(all_vertices - outer_center, normal)
            # The side length s is related to the max projection distance from the center
            # For a regular hexagon with side s, the apothem is s * sqrt(3)/2.
            # The vertices of the inner hexagons must lie within the outer hexagon.
            # The distance from center to side is the apothem.
            # So max projection must be <= apothem = s * sqrt(3)/2.
            # Thus s >= max_projection / (sqrt(3)/2).
            current_s = np.max(projections) / (np.sqrt(3) / 2)
            if current_s > max_projection: # We want the max of the required s
                max_projection = current_s
                # Note: we are storing the required s here, not the projection. 
                # Let's rename variable for clarity in logic, but keep code structure.
        
        return max_projection, outer_center

    def objective(params):
        if time.time() > deadline - 30:
            return 1e10
        cluster_angle, outer_angle, ocx, ocy = params
        cluster_angle_rad = np.radians(cluster_angle)
        
        cos_a = np.cos(cluster_angle_rad)
        sin_a = np.sin(cluster_angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_centers = centers @ rotation_matrix.T
        
        outer_center = np.array([ocx, ocy])
        s, _ = compute_enclosing_hex_side(rotated_centers, angles, outer_center, outer_angle)
        return s
    
    best_s = np.inf
    best_params = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Initial guess: cluster angle 0, outer angle 0, outer center at cluster centroid
    centroid = np.mean(centers, axis=0)
    
    starts = [
        np.array([0, 0, centroid[0], centroid[1]]),
        np.array([30, 0, centroid[0], centroid[1]]),
        np.array([0, 30, centroid[0], centroid[1]]),
        np.array([15, 15, centroid[0], centroid[1]]),
        np.array([45, 0, centroid[0], centroid[1]]),
        np.array([0, 45, centroid[0], centroid[1]]),
        np.array([60, 0, centroid[0], centroid[1]]),
        np.array([10, 20, centroid[0], centroid[1]]),
        # Try some offsets for outer center
        np.array([0, 0, centroid[0]+0.1, centroid[1]+0.1]),
        np.array([0, 0, centroid[0]-0.1, centroid[1]-0.1]),
    ]
    
    for start in starts:
        if time.time() > deadline - 60:
            break
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    cluster_angle, outer_angle, ocx, ocy = best_params
    cluster_angle_rad = np.radians(cluster_angle)
    
    cos_a = np.cos(cluster_angle_rad)
    sin_a = np.sin(cluster_angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_centers = centers @ rotation_matrix.T
    
    outer_center = np.array([ocx, ocy])
    s, _ = compute_enclosing_hex_side(rotated_centers, angles, outer_center, outer_angle)
    
    inner_hex_data = np.column_stack([rotated_centers, angles])
    outer_hex_data = np.array([ocx, ocy, outer_angle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
