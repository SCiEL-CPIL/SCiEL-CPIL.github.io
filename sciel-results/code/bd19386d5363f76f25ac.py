# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    # Hexagonal lattice arrangement with slight spacing increase for separation
    sqrt3 = np.sqrt(3)
    d = 2.0 + 2e-5  # spacing between adjacent hexagon centers
    
    # Arrangement: 3+3+3+2 pattern in hexagonal lattice
    # Basis vectors: a1 = (d, 0), a2 = (d/2, d*sqrt3/2)
    a1 = np.array([d, 0.0])
    a2 = np.array([d/2, d*sqrt3/2])
    
    # Define positions in lattice coordinates (i, j) -> i*a1 + j*a2
    lattice_coords = [
        (0, 0), (1, 0), (2, 0),      # Row 0
        (0, 1), (1, 1), (2, 1),      # Row 1
        (0, 2), (1, 2), (2, 2),      # Row 2
        (0, 3), (1, 3),              # Row 3
    ]
    
    centers = np.array([i*a1 + j*a2 for i, j in lattice_coords])
    
    # All inner hexagons have angle 0
    inner_angles = np.zeros(11)
    
    # Compute all vertices of all inner hexagons
    all_verts = []
    for i in range(11):
        cx, cy = centers[i]
        angles = np.radians(60 * np.arange(6))
        verts = np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Center of mass
    com = np.mean(all_verts, axis=0)
    
    def enclosing_side(angle_deg, center_offset):
        """Compute minimum side length of regular hexagon."""
        oa = np.radians(angle_deg)
        verts = all_verts - com - center_offset
        
        min_s = 0.0
        for i in range(6):
            a_i = oa + np.radians(60 * i)
            a_ip1 = oa + np.radians(60 * (i + 1))
            e_unit = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
            crosses = verts[:, 0] * e_unit[1] - verts[:, 1] * e_unit[0]
            min_cross = np.min(crosses)
            required_s = -2 * min_cross / np.sqrt(3)
            if required_s > min_s:
                min_s = required_s
        
        return max(min_s, np.sqrt(11))
    
    # Optimize angle with center at COM
    def obj_angle(angle_deg):
        return enclosing_side(angle_deg, np.array([0.0, 0.0]))
    
    best_s = np.inf
    best_angle = 0.0
    best_center = np.array([0.0, 0.0])
    
    result = minimize_scalar(obj_angle, bounds=(0, 60), method='bounded', options={'xatol': 1e-10})
    if result.fun < best_s:
        best_s = result.fun
        best_angle = result.x
    
    # Optimize center offset too
    def obj_full(params):
        return enclosing_side(params[0], np.array([params[1], params[2]]))
    
    x0 = np.array([best_angle, 0.0, 0.0])
    result2 = minimize(obj_full, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    if result2.fun < best_s:
        best_s = result2.fun
        best_angle = result2.x[0]
        best_center = result2.x[1:3]
    
    outer_center = com + best_center
    outer_angle = best_angle
    side = max(best_s, np.sqrt(11))
    
    inner_hex_data = np.column_stack([centers, inner_angles])
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, float(side)

# EVOLVE-BLOCK-END
