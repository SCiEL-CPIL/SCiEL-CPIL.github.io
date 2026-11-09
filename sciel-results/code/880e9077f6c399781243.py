# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))

def _hex_edge_normals(vertices):
    n = 6
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normals[i] = normal / norm
    return normals

def _separation_check(v1, v2):
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _is_inside_outer(inner_vertices, outer_vertices):
    n_out = len(outer_vertices)
    for v in inner_vertices:
        inside = True
        for i in range(n_out):
            p1 = outer_vertices[i]
            p2 = outer_vertices[(i + 1) % n_out]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                inside = False
                break
        if not inside:
            return False
    return True

def _compute_s(inner_positions, outer_center, outer_angle_deg, inner_angles):
    outer_center = np.array(outer_center)
    inner_centers = np.array(inner_positions)
    shifted = inner_centers - outer_center
    
    a = np.radians(outer_angle_deg)
    normal_angles = a + np.radians(60) * np.arange(6) + np.radians(30)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    all_vertices = []
    for i in range(len(inner_centers)):
        cx, cy = shifted[i]
        verts = _hex_vertices(cx, cy, 1, inner_angles[i])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    projections = all_vertices @ normals.T
    max_proj = projections.max()
    
    s = 2 * max_proj / np.sqrt(3)
    return s

def _obj_positions(params_pos, inner_angles):
    pos = params_pos.reshape(12, 2)
    
    penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(pos[i,0], pos[i,1], 1, inner_angles[i])
            v2 = _hex_vertices(pos[j,0], pos[j,1], 1, inner_angles[j])
            
            n1 = _hex_edge_normals(v1)
            n2 = _hex_edge_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            min_sep = np.inf
            for normal in all_normals:
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                
                if max1 < min2:
                    sep = min2 - max1
                elif max2 < min1:
                    sep = min1 - max2
                else:
                    sep = -min(max1 - min2, max2 - min1)
                
                min_sep = min(min_sep, sep)
            
            if min_sep < 0:
                penalty += 100 * min_sep * min_sep
    
    if penalty > 50:
        return 1000 + penalty
    
    # Fit outer hex: optimize center (2) and angle (1)
    def obj_outer(params_outer):
        oc = params_outer[:2]
        oa = params_outer[2]
        return _compute_s(pos, oc, oa, inner_angles)
    
    oc_init = np.mean(pos, axis=0)
    oa_init = 0
    try:
        res_outer = minimize(obj_outer, np.concatenate([oc_init, [oa_init]]), method='Nelder-Mead', 
                             options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 300})
        s_val = res_outer.fun
    except:
        s_val = _compute_s(pos, oc_init, 0, inner_angles)
    
    return s_val + penalty

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    # Initial seed
    init_pos = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17],
        [0, -4]
    ])
    init_angles = np.zeros(12)
    
    best_pos = init_pos.copy()
    best_angles = init_angles.copy()
    best_oc = np.array([0, 0])
    best_oa = 0
    best_s = 8.0
    
    # Evaluate seed
    def eval_full(pos, angles):
        def obj_outer(params_outer):
            oc = params_outer[:2]
            oa = params_outer[2]
            return _compute_s(pos, oc, oa, angles)
        
        oc_init = np.mean(pos, axis=0)
        oa_init = 0
        try:
            res_outer = minimize(obj_outer, np.concatenate([oc_init, [oa_init]]), method='Nelder-Mead', 
                                 options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
            s_val = res_outer.fun
            oc = res_outer.x[:2]
            oa = res_outer.x[2]
        except:
            s_val = _compute_s(pos, oc_init, 0, angles)
            oc = oc_init
            oa = 0
        
        # Check validity
        valid = True
        for i in range(12):
            for j in range(i+1, 12):
                v1 = _hex_vertices(pos[i,0], pos[i,1], 1, angles[i])
                v2 = _hex_vertices(pos[j,0], pos[j,1], 1, angles[j])
                if not _separation_check(v1, v2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            outer_verts = _hex_vertices(oc[0], oc[1], s_val, oa)
            for i in range(12):
                inner_verts = _hex_vertices(pos[i,0], pos[i,1], 1, angles[i])
                if not _is_inside_outer(inner_verts, outer_verts):
                    valid = False
                    break
        
        if valid:
            return s_val, oc, oa, True
        else:
            return s_val, oc, oa, False
    
    s_seed, oc_seed, oa_seed, valid_seed = eval_full(init_pos, init_angles)
    if valid_seed:
        best_s = s_seed
        best_oc = oc_seed
        best_oa = oa_seed
    
    # Optimization loop
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if r == 0:
            pos_init = init_pos.copy()
            angles_init = init_angles.copy()
        else:
            pos_init = init_pos + np.random.normal(0, 0.5, init_pos.shape)
            angles_init = np.random.uniform(0, 60, 12)
        
        try:
            res = minimize(_obj_positions, pos_init.flatten(), args=(angles_init,), method='Nelder-Mead',
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000, 'maxfev': 5000})
            
            pos_candidate = res.x.reshape(12, 2)
            
            # Refine angles? The current objective fixes angles.
            # For a better score, we should maybe optimize angles too.
            # But 24+12=36 vars is heavy for NM.
            # Let's try to optimize angles in a second pass for the best position found.
            
            s_cand, oc_cand, oa_cand, valid_cand = eval_full(pos_candidate, angles_init)
            
            if valid_cand and s_cand < best_s:
                best_s = s_cand
                best_pos = pos_candidate
                best_angles = angles_init
                best_oc = oc_cand
                best_oa = oa_cand
                
        except Exception as e:
            pass
        
        # Check deadline
        if time.time() > deadline - 30:
            break
            
        # Optional: Local refinement of angles for the best solution found so far
        # This is a structural change: treating angles as variables in a nested loop.
        # Given time constraints, we'll do a simple grid search or local search on angles for the current best.
        if r > 0 and (r % 3 == 0): # Every 3rd iteration, try to refine angles
            if time.time() > deadline - 60:
                break
            
            # Try small perturbations to angles of best_pos
            improved = False
            for _ in range(5):
                angles_pert = best_angles + np.random.uniform(-5, 5, 12)
                angles_pert = angles_pert % 60
                s_p, oc_p, oa_p, valid_p = eval_full(best_pos, angles_pert)
                if valid_p and s_p < best_s:
                    best_s = s_p
                    best_angles = angles_pert
                    best_oc = oc_p
                    best_oa = oa_p
                    improved = True
            if not improved:
                pass

    # Final validation
    final_s, final_oc, final_oa, final_valid = eval_full(best_pos, best_angles)
    
    if not final_valid:
        # Fallback to seed if final is invalid
        final_s, final_oc, final_oa, _ = eval_full(init_pos, init_angles)
        best_pos = init_pos
        best_angles = init_angles

    # Construct return values
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_pos[i, 0]
        inner_hex_data[i, 1] = best_pos[i, 1]
        inner_hex_data[i, 2] = best_angles[i]
    
    outer_hex_data = np.array([final_oc[0], final_oc[1], final_oa])
    
    return inner_hex_data, outer_hex_data, final_s

# EVOLVE-BLOCK-END
