import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    
    # 1. Generate 19 centers of k=3 hex cluster
    centers_19 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                x = np.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                centers_19.append([x, y])
    centers_19 = np.array(centers_19)
    
    # 2. Generate all subsets of 12
    indices = np.arange(19)
    best_s = float('inf')
    best_subset = None
    best_params = None
    
    candidates = []
    # Precompute norms for speed?
    for comb in combinations(indices, 12):
        pts = centers_19[list(comb)]
        com = np.mean(pts, axis=0)
        pts_centered = pts - com
        max_norm = np.max(np.linalg.norm(pts_centered, axis=1))
        s_est = max_norm * 2 / np.sqrt(3) + 1
        candidates.append((s_est, comb))
        
    candidates.sort(key=lambda x: x[0])
    
    # Top 100 candidates
    top_candidates = candidates[:100]
    
    for s_est, comb in top_candidates:
        if time.time() - start_time > 500:
            break
            
        pts = centers_19[list(comb)]
        
        def objective(params):
            dx, dy, theta = params
            angles = theta + np.arange(6) * np.pi / 3
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            projections = pts @ normals.T
            c_projections = np.array([dx, dy]) @ normals.T
            
            max_projections = np.max(projections, axis=0) - c_projections
            return np.max(max_projections)
        
        com = np.mean(pts, axis=0)
        x0 = [com[0], com[1], 0.0]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.success:
                dx, dy, theta = res.x
                max_val = res.fun
                s = 2 / np.sqrt(3) * max_val
                if s < best_s:
                    best_s = s
                    best_subset = list(comb)
                    best_params = (dx, dy, theta)
        except Exception:
            pass

    if best_s == float('inf'):
        # Fallback to seed-like result
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    pts = centers_19[best_subset]
    c = np.array([best_params[0], best_params[1]])
    inner_centers = pts - c
    
    inner_data = np.zeros((12, 3))
    for i in range(12):
        inner_data[i] = [inner_centers[i][0], inner_centers[i][1], 0]
        
    outer_data = np.array([0, 0, np.degrees(best_params[2])])
    
    return inner_data, outer_data, best_s
