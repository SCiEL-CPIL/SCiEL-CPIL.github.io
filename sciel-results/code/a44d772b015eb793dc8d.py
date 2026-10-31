import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    def exact_score(pts):
        # pts: (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    def prepare_params(pts):
        # Find pair with min distance to fix? Or just fix first two?
        # Fixing arbitrary points might lead to bad local minima if they are far apart.
        # Better to find the pair with minimum distance and fix them to (0,0) and (1,0).
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_idx = np.argmin(d_sq)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            # Degenerate, return random
            return None
        
        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        # Transform all points
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        # Reorder so i0 is index 0, i1 is index 1
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # Check if transformed[0] is (0,0) and transformed[1] is (1,0)
        # Due to floating point, might be slightly off, but should be close.
        # We will optimize the remaining 14 points (28 params).
        return transformed[2:].flatten(), perm

    def params_to_pts(params, perm):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        # Reverse permutation
        inv_perm = np.argsort(perm)
        return pts[inv_perm]

    def soft_objective(params, beta=50.0):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Avoid zero
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
        
        return soft_max_d_sq / soft_min_d_sq

    def log_soft_objective(params, beta=50.0):
        """
        Minimizing log(max/min) is equivalent to minimizing log(max) - log(min).
        This can be more numerically stable for optimization.
        """
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Avoid zero
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max value
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min value
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8 or soft_max_d_sq < 1e-8:
            return 1e10
            
        return np.log(soft_max_d_sq) - np.log(soft_min_d_sq)

    best_score = 1e10
    best_params = None
    best_perm = None
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Random seeds
    for _ in range(3):
        pts = np.random.uniform(-1, 1, (N, 2))
        seeds.append(pts)
        
    # 2. Hexagonal lattice seed (perturbed)
    hex_pts = []
    R = 4
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Take first 16
    hex_seed = hex_pts[:N].copy()
    hex_seed += np.random.normal(0, 0.1, hex_seed.shape)
    seeds.append(hex_seed)
    
    # 3. Concentric circles seed
    circ_pts = []
    # 1 center + 6 ring 1 + 9 ring 2? No, 1+6=7, 7+9=16.
    # Ring 1: radius 1, 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        circ_pts.append([np.cos(angle), np.sin(angle)])
    # Ring 2: radius 2, 9 points? Or radius 1.5?
    # Let's try radius 1.5 with 9 points
    for i in range(9):
        angle = 2 * np.pi * i / 9
        circ_pts.append([1.5 * np.cos(angle), 1.5 * np.sin(angle)])
    circ_pts.append([0,0])
    circ_pts = np.array(circ_pts)
    circ_seed = circ_pts.copy()
    circ_seed += np.random.normal(0, 0.1, circ_seed.shape)
    seeds.append(circ_seed)

    # 4. Grid seed
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([i, j])
    grid_seed = np.array(grid_pts, dtype=float)
    grid_seed += np.random.normal(0, 0.05, grid_seed.shape)
    seeds.append(grid_seed)

    num_restarts = len(seeds)
    # We have ~140s. Let's spend ~15s per seed for optimization, plus polish.
    # 5 seeds * 15s = 75s. Plus polish 20s. Total 95s. Safe.
    
    for r in range(num_restarts):
        if time.time() > deadline - 30: # Leave time for final polish
            break
            
        pts = seeds[r]
        
        # Ensure distinctness
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-4:
                    pts[j] += np.random.uniform(-0.1, 0.1, 2)
        
        res_prep = prepare_params(pts)
        if res_prep is None:
            continue
        init_params, perm = res_prep
        
        # Optimization 1: Powell (Derivative-free, robust for geometric problems)
        # Using log objective for stability
        bnds = [(-5, 5)] * 28
        
        try:
            # Powell can be slow, so limit function evaluations or iterations
            res = minimize(log_soft_objective, init_params, method='Powell', bounds=bnds,
                           options={'maxiter': 100, 'ftol': 1e-8, 'xtol': 1e-6})
            
            candidate_params = res.x
            candidate_pts = params_to_pts(candidate_params, perm)
            candidate_score = exact_score(candidate_pts)
            
            if candidate_score < best_score:
                best_score = candidate_score
                best_params = candidate_params
                best_perm = perm
                
            # Optimization 2: L-BFGS-B polish on the result of Powell
            # This leverages the gradient for fine-tuning
            if time.time() < deadline - 25:
                res2 = minimize(soft_objective, candidate_params, method='L-BFGS-B', bounds=bnds,
                                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
                
                candidate_params2 = res2.x
                candidate_pts2 = params_to_pts(candidate_params2, perm)
                candidate_score2 = exact_score(candidate_pts2)
                
                if candidate_score2 < best_score:
                    best_score = candidate_score2
                    best_params = candidate_params2
                    best_perm = perm
                    
        except Exception as e:
            pass

    # Final Polish on the best result found
    if best_params is not None and time.time() < deadline - 5:
        # Use a higher beta for sharper approximation to min/max
        beta_final = 100.0
        
        def soft_objective_final(params):
            return soft_objective(params, beta=beta_final)
        
        bnds = [(-5, 5)] * 28
        try:
            res_final = minimize(soft_objective_final, best_params, method='L-BFGS-B', bounds=bnds, 
                                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() < deadline - 2:
                candidate_params = res_final.x
                candidate_pts = params_to_pts(candidate_params, best_perm)
                candidate_score = exact_score(candidate_pts)
                
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_params = candidate_params
                    
        except Exception:
            pass

    # Construct final points
    if best_params is None:
        # Fallback: regular hexagonal packing
        pts = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                pts.append([x, y])
        pts = np.array(pts[:N])
        return pts

    final_pts = params_to_pts(best_params, best_perm)
    
    # One last check: maybe the points are not distinct?
    # The problem requires distinct points.
    # If min dist is 0, score is inf.
    # We should verify.
    
    return final_pts
