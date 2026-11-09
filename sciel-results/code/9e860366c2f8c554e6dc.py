import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600
    
    # Normals for the hexagon edges (angles 30, 90, 150)
    angles = np.deg2rad([30, 90, 150])
    normals = np.array([
        [np.cos(a), np.sin(a)] for a in angles
    ])
    
    # Vertices of a unit hexagon (side 1)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([
        [np.cos(a), np.sin(a)] for a in v_angles
    ])
    
    # Threshold for non-overlap (distance between centers projected on normal)
    # Forbidden zone is hexagon of side 2. Distance to edge is sqrt(3).
    # We add a small margin to ensure strict separation > 1e-6
    threshold = np.sqrt(3) + 1e-5

    def objective_and_penalty(x):
        pts = x.reshape(11, 2)
        
        # Calculate S_req
        # S_req = (2/sqrt(3)) * max_i max_v max_k ( |(pts[i] + hex_verts[v]) dot normals[k]| )
        
        # pts: (11, 2)
        # hex_verts: (6, 2)
        # We need to compute projections of all (pts[i] + hex_verts[v]) onto normals.
        
        # Efficient calculation:
        # For each normal k, compute max over i, v of | (pts[i] + hex_verts[v]) dot n_k |
        # This is max over i, v of | pts[i] dot n_k + hex_verts[v] dot n_k |
        
        max_proj_all = 0.0
        for k in range(3):
            n_k = normals[k]
            # pts_proj: (11,)
            pts_proj = pts @ n_k
            # verts_proj: (6,)
            verts_proj = hex_verts @ n_k
            
            # We want max_{i, v} | pts_proj[i] + verts_proj[v] |
            # This is equivalent to max( max_i pts_proj[i] + max_v verts_proj[v], -min_i pts_proj[i] - min_v verts_proj[v] )?
            # No. | A + B | max is not simply sum of max abs.
            # But we can just compute the 11*6 sums? 66 operations per normal. 3 normals. 198 ops. Very fast.
            
            # Actually, we can vectorize.
            # sums: (11, 6)
            sums = pts_proj[:, np.newaxis] + verts_proj[np.newaxis, :]
            max_abs_sum = np.max(np.abs(sums))
            
            if max_abs_sum > max_proj_all:
                max_proj_all = max_abs_sum
                
        S_req = (2.0 / np.sqrt(3.0)) * max_proj_all
        
        # Calculate Penalty for overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj_v = np.abs(v @ normals.T)
                max_proj_v = np.max(proj_v)
                if max_proj_v < threshold:
                    depth = threshold - max_proj_v
                    penalty += depth * depth
                    
        return S_req + 100.0 * penalty

    # Initial guess
    d = np.sqrt(3)
    pts = []
    pts.append([0, 0])
    for k in range(6):
        angle = np.deg2rad(60 * k)
        pts.append([d * np.cos(angle), d * np.sin(angle)])
        
    spike_angles = [0, 120, 180, 300]
    for angle_deg in spike_angles:
        angle = np.deg2rad(angle_deg)
        pts.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
        
    x0 = np.array(pts).flatten()
    
    best_x = x0
    best_val = objective_and_penalty(x0)
    
    # Optimization
    res = minimize(objective_and_penalty, x0, method='Powell', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if res.fun < best_val:
        best_x = res.x
        best_val = res.fun
        
    # Check time and maybe run more
    if time.time() < deadline - 60:
         # Try a second run with slightly perturbed start
         x0_pert = x0 + np.random.normal(0, 0.1, x0.shape)
         res2 = minimize(objective_and_penalty, x0_pert, method='Powell', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
         if res2.fun < best_val:
             best_x = res2.x
             best_val = res2.fun

    pts_final = best_x.reshape(11, 2)
    
    # Recalculate S for final result
    max_proj_all = 0.0
    for k in range(3):
        n_k = normals[k]
        pts_proj = pts_final @ n_k
        verts_proj = hex_verts @ n_k
        sums = pts_proj[:, np.newaxis] + verts_proj[np.newaxis, :]
        max_abs_sum = np.max(np.abs(sums))
        if max_abs_sum > max_proj_all:
            max_proj_all = max_abs_sum
            
    side = (2.0 / np.sqrt(3.0)) * max_proj_all
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pts_final[:, 0]
    inner_hex_data[:, 1] = pts_final[:, 1]
    inner_hex_data[:, 2] = 0
    
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, side
