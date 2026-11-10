import numpy as np
from scipy.optimize import minimize_scalar


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    using a hexagonal lattice arrangement and optimizing the rotation to minimize the outer hexagon size.
    """
    sqrt3 = np.sqrt(3)
    
    # Define 12 positions in a compact hexagonal lattice arrangement using axial coordinates
    axial_coords = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (-1, 1),
        (1, -1),
        (2, 0),
        (-2, 0),
        (1, 1),
        (1, -2),
        (-1, 2),
    ]
    
    # Convert axial to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    centers = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)
    
    # Center the configuration at origin
    center_offset = np.mean(centers, axis=0)
    centers = centers - center_offset
    
    # Function to compute minimal outer hexagon side for a given inner rotation
    def compute_outer_side(inner_rotation_deg):
        theta = np.radians(inner_rotation_deg)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        
        rotated_centers = centers @ rot_matrix.T
        
        # Compute all vertices of all inner hexagons (each has angle = inner_rotation_deg)
        all_vertices = []
        for cx, cy in rotated_centers:
            for i in range(6):
                angle = theta + np.radians(60 * i)
                vx = cx + np.cos(angle)
                vy = cy + np.sin(angle)
                all_vertices.append([vx, vy])
        all_vertices = np.array(all_vertices)
        
        # Find minimal regular hexagon orientation and apothem
        def apothem_for_orientation(phi):
            max_val = 0.0
            for k in range(6):
                angle = phi + np.radians(60 * k)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                projections = np.abs(all_vertices[:, 0] * cos_a + all_vertices[:, 1] * sin_a)
                max_val = max(max_val, float(np.max(projections)))
            return max_val
        
        result = minimize_scalar(apothem_for_orientation, bounds=(0.0, np.pi / 3), method='bounded',
                                 options={'xatol': 1e-6})
        min_apothem = result.fun
        s = 2.0 * min_apothem / sqrt3
        return s, result.x  # return side length and optimal outer orientation
    
    # Optimize inner rotation
    def objective(inner_rot):
        s, _ = compute_outer_side(inner_rot)
        return s
    
    result_inner = minimize_scalar(objective, bounds=(0.0, 180.0), method='bounded',
                                   options={'xatol': 1e-4})
    best_inner_rot = result_inner.x
    best_s, best_outer_rot_rad = compute_outer_side(best_inner_rot)
    best_outer_rot_deg = np.degrees(best_outer_rot_rad)
    
    # Construct final inner hexagon data
    theta = np.radians(best_inner_rot)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    final_centers = centers @ rot_matrix.T
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = best_inner_rot
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_rot_deg])
    
    return inner_hex_data, outer_hex_data, best_s
