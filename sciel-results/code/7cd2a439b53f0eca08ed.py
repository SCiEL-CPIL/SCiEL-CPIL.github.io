import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    sqrt3 = np.sqrt(3.0)
    
    def get_inner_vertices(center, angle):
        angles = angle + np.arange(6) * np.pi / 3.0
        return center + np.column_stack([np.cos(angles), np.sin(angles)])
    
    def point_in_hexagon_violation(p, center, orientation, side):
        d = p - center
        angles = orientation + np.arange(6) * np.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        apothem = side * sqrt3 / 2.0
        violations = normals @ d - apothem
        return np.max(violations)
    
    def penalty_objective(x):
        inner_centers = x[0:24].reshape(12, 2)
        inner_angles = x[24:36]
        outer_center = x[36:38]
        outer_angle = x[38]
        outer_side = x[39]
        
        if outer_side < 0.5:
            return 1e6
        
        containment_penalty = 0.0
        for i in range(12):
            verts = get_inner_vertices(inner_centers[i], inner_angles[i])
            for j in range(6):
                viol = point_in_hexagon_violation(verts[j], outer_center, outer_angle, outer_side)
                if viol > 0:
                    containment_penalty += viol ** 2
        
        overlap_penalty = 0.0
        min_dist = sqrt3
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(inner_centers[i] - inner_centers[j])
                if d < min_dist:
                    overlap_penalty += (min_dist - d) ** 2
        
        return outer_side + 100.0 * (containment_penalty + overlap_penalty)
    
    def hex_to_cart(q, r):
        x = 1.5 * q
        y = sqrt3 / 2 * q + sqrt3 * r
        return x, y
    
    hex_coords = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2),
        (0, 3), (1, 3)
    ]
    
    x0 = np.zeros(40)
    for i, (q, r) in enumerate(hex_coords):
        x, y = hex_to_cart(q, r)
        x0[2*i] = x
        x0[2*i+1] = y
    x0[24:36] = 0.0
    x0[36] = 1.5
    x0[37] = 1.5
    x0[38] = 0.0
    x0[39] = 6.0
    
    result = minimize(penalty_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
    
    x = result.x
    inner_centers = x[0:24].reshape(12, 2)
    inner_angles_deg = np.degrees(x[24:36])
    outer_center = (float(x[36]), float(x[37]))
    outer_angle_deg = float(np.degrees(x[38]))
    outer_side = float(x[39])
    
    inner = [[float(inner_centers[i][0]), float(inner_centers[i][1]), float(inner_angles_deg[i])] 
             for i in range(12)]
    
    return inner, outer_center, outer_side, outer_angle_deg
