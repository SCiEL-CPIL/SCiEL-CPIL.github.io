import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2
EPS_FD = 1e-4  # Finite difference step size

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

    def finite_diff_gradient(params, beta=50.0):
        """
        Compute gradient of log_soft_objective using central finite differences.
        Returns gradient vector of same shape as params.
        """
        grad = np.zeros_like(params)
        f0 = log_soft_objective(params, beta)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += EPS_FD
            params_minus[i] -= EPS_FD
            
            f_plus = log_soft_objective(params_plus, beta)
            f_minus = log_soft_objective(params_minus, beta)
            
            # If either evaluation fails (returns 1e10), use forward/backward difference
            if f_plus > 1e9:
                f_plus = f0 + 2 * (f_minus - f0) if f_minus < 1e9 else f0
            if f_minus > 1e9:
                f_minus = f0 + 2 * (f_plus - f0) if f_plus < 1e9 else f0
            
            grad[i] = (f_plus - f_minus) / (2 * EPS_FD)
        
        return grad

    def gradient_flow_optimize(params, beta=50.0, max_iter=200, lr=0.01, time_limit=15.0):
        """
        Perform gradient flow optimization using finite-difference gradients.
        This is a simple but effective local refinement method.
        """
        start_time = time.time()
        current_params = params.copy()
        current_obj = log_soft_objective(current_params, beta)
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Compute gradient
            grad = finite_diff_gradient(current_params, beta)
            
            # Adaptive learning rate: reduce if objective increases
            new_params = current_params - lr * grad
            
            new_obj = log_soft_objective(new_params, beta)
            
            if new_obj < current_obj:
                # Accept the step
                current_params = new_params
                current_obj = new_obj
                # Slightly increase learning rate if improving steadily
                lr = min(lr * 1.05, 0.1)
            else:
                # Reject and reduce learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        return current_params, current_obj

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
    # Generate a larger hexagonal lattice and pick the 16 points closest to the center
    # This creates a compact cluster which is likely to have a good ratio
    hex_pts = []
    R = 5
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Sort by distance to origin
    dists_to_origin = np.linalg.norm(hex_pts, axis=1)
    sorted_indices = np.argsort(dists_to_origin)
    hex_seed = hex_pts[sorted_indices[:N]].copy()
    # Center it
    hex_seed -= np.mean(hex_seed, axis=0)
    hex_seed += np.random.normal(0, 0.05, hex_seed.shape)
    seeds.append(hex_seed)
    
    # 3. Concentric circles seed (1 center, 6 in ring 1, 9 in ring 2)
    circ_pts = []
    circ_pts.append([0,0])
    # Ring 1: radius 1, 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        circ_pts.append([np.cos(angle), np.sin(angle)])
    # Ring 2: radius approx 1.8, 9 points
    # Optimal radius for 9 points around 6 points? 
    # Distance from center to ring 2 points should be > 1 (dist to ring 1)
    # Distance between ring 2 points should be > 1
    # Circumference 2*pi*r >= 9*1 => r >= 9/(2*pi) ~ 1.43
    # Distance from ring 1 to ring 2:
    # If angles align, dist is r2-r1. If offset, dist is sqrt(r1^2+r2^2-2r1r2 cos(delta))
    # Let's try r2 = 1.8
    r2 = 1.8
    for i in range(9):
        angle = 2 * np.pi * i / 9 + np.pi/9 # Offset to interleave
        circ_pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    circ_pts = np.array(circ_pts)
    circ_seed = circ_pts.copy()
    circ_seed += np.random.normal(0, 0.05, circ_seed.shape)
    seeds.append(circ_seed)

    # 4. Grid seed (4x4)
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([i, j])
    grid_seed = np.array(grid_pts, dtype=float)
    grid_seed -= np.mean(grid_seed, axis=0)
    grid_seed += np.random.normal(0, 0.05, grid_seed.shape)
    seeds.append(grid_seed)
    
    # 5. Another hexagonal variation: 3 rings of hexagon?
    # Center (1), Ring 1 (6), Ring 2 (12) -> Total 19. We need 16.
    # Take center + ring 1 (6) + 9 from ring 2?
    hex_pts2 = []
    hex_pts2.append([0,0])
    for i in range(6):
        angle = 2 * np.pi * i / 6
        hex_pts2.append([np.cos(angle), np.sin(angle)])
    # Ring 2 has 12 points. We need 9 more.
    # Pick 9 consecutive points from ring 2 to minimize diameter?
    # Or pick them to maximize min distance?
    # Let's just take the first 9 from ring 2 (angles 0 to 150 deg)
    for i in range(9):
        angle = 2 * np.pi * i / 12
        hex_pts2.append([2 * np.cos(angle), 2 * np.sin(angle)])
    hex_pts2 = np.array(hex_pts2)
    hex_seed2 = hex_pts2.copy()
    hex_seed2 += np.random.normal(0, 0.05, hex_seed2.shape)
    seeds.append(hex_seed2)

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
        
        bnds = [(-5, 5)] * 28
        
        try:
            # Optimization 1: Gradient flow using finite-difference gradients
            # This is the primary refinement strategy
            gf_params, gf_obj = gradient_flow_optimize(
                init_params, beta=50.0, max_iter=300, lr=0.02, time_limit=12.0
            )
            
            candidate_pts = params_to_pts(gf_params, perm)
            candidate_score = exact_score(candidate_pts)
            
            if candidate_score < best_score:
                best_score = candidate_score
                best_params = gf_params.copy()
                best_perm = perm.copy()
            
            # Optimization 2: L-BFGS-B polish on the result of gradient flow
            # This leverages the smooth surrogate for fine-tuning
            if time.time() < deadline - 20:
                res2 = minimize(log_soft_objective, gf_params, method='L-BFGS-B', bounds=bnds,
                                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                
                candidate_params2 = res2.x
                candidate_pts2 = params_to_pts(candidate_params2, perm)
                candidate_score2 = exact_score(candidate_pts2)
                
                if candidate_score2 < best_score:
                    best_score = candidate_score2
                    best_params = candidate_params2.copy()
                    best_perm = perm.copy()
                    
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
