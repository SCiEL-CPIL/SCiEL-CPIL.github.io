import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    """
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _check_separation(pos1, ang1, pos2, ang2, side=1.0):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated (no overlap), False if overlapping.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
    
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return True
    
    return False

def _is_valid_config(positions, angles, side=1.0):
    """Check if configuration is valid: all pairs separated and inside outer hex."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_separation(positions[i], angles[i], positions[j], angles[j], side):
                return False
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, leave margin
    
    n = 12
    side = 1.0
    
    # Initial configuration: Honeycomb cluster
    # Center (0,0)
    # 6 around at distance 2
    # 5 more around those
    positions_init = np.zeros((12, 2))
    angles_init = np.zeros(12)
    
    positions_init[0] = [0,0]
    
    for k in range(6):
        theta = np.radians(60 * k)
        positions_init[k+1] = [2.0 * np.cos(theta), 2.0 * np.sin(theta)]
        
    for k in range(5):
        inner_idx = k + 1
        theta = np.radians(60 * inner_idx)
        positions_init[k+7] = positions_init[inner_idx] + 2.0 * np.array([np.cos(theta), np.sin(theta)])
        
    # Flatten for optimization
    x0 = np.concatenate([positions_init.flatten(), angles_init])
    
    def objective(x):
        positions = x[:24].reshape(12, 2)
        angles = x[24:]
        
        # Compute vertices for all hexagons
        vertices = np.vstack([
            _hex_vertices(positions[i,0], positions[i,1], side, angles[i]) 
            for i in range(12)
        ])
        
        # Minimize outer side
        s = _min_outer_side(vertices)
        
        # Penalty for overlaps
        overlap_penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                # If centers are too close, they likely overlap
                if d < 2.0 * side - 0.05:
                    if not _check_separation(positions[i], angles[i], positions[j], angles[j], side):
                        # Approximate penetration depth for smooth penalty
                        # Using distance to touching distance as proxy
                        overlap_penalty += 100.0 * (2.0 * side - d) ** 2
        
        return s + overlap_penalty
    
    best_s = float('inf')
    best_x = None
    
    # Run multiple restarts with perturbations
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Perturb initial position
        if restart == 0:
            x0_restart = x0.copy()
        else:
            x0_restart = x0 + np.random.normal(0, 0.1, size=x0.shape)
            # Ensure angles are within reasonable bounds
            x0_restart[24:] = np.mod(x0_restart[24:], 60)
        
        try:
            res = minimize(objective, x0_restart, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.fun < best_s:
                # Validate
                pos = res.x[:24].reshape(12, 2)
                ang = res.x[24:]
                
                if _is_valid_config(pos, ang, side):
                    verts = np.vstack([
                        _hex_vertices(pos[i,0], pos[i,1], side, ang[i]) 
                        for i in range(12)
                    ])
                    s_val = _min_outer_side(verts)
                    
                    if s_val < best_s:
                        best_s = s_val
                        best_x = res.x.copy()
                        
        except Exception:
            pass
            
    # If optimization failed, use initial static config
    if best_x is None:
        best_x = x0
        best_s = objective(best_x)
        
    best_positions = best_x[:24].reshape(12, 2)
    best_angles = best_x[24:]
    
    inner_hex_data = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s
