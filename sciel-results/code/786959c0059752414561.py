import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Generate hexagonal lattice points
    R = 8
    pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            pts.append((x, y))
    
    pts = np.array(pts)
    num_pts = len(pts)
    
    # Get initial subset: 16 closest to origin
    dists_from_origin = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N]
    
    # Candidate pool for swaps
    pool_size = 200
    candidate_indices = sorted_indices[:pool_size]
    
    def score_subset(indices):
        sub_pts = pts[indices]
        diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2
    
    current_indices = initial_subset.copy()
    current_score = score_subset(current_indices)
    
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    # Local search: swap points to improve
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i_idx in range(N):
            if time.time() > deadline - 30:
                break
            for j in outside_candidates:
                if time.time() > deadline - 30:
                    break
                new_indices = current_indices.copy()
                new_indices[i_idx] = j
                new_score = score_subset(new_indices)
                if new_score < current_score:
                    current_score = new_score
                    current_indices = new_indices
                    in_subset = set(current_indices)
                    outside_candidates = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    break
            if improved:
                break
    
    best_pts = pts[current_indices]
    
    # Normalize: fix closest pair to (0,0) and (1,0)
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair
    if i0 > i1:
        i0, i1 = i1, i0
    
    p0 = best_pts[i0]
    p1 = best_pts[i1]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return best_pts
    
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (best_pts - p0) @ R_mat.T / norm_v
    
    # Reorder so i0 is at index 0, i1 at index 1
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    def exact_objective(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-10:
            return 1e10
        return max_d_sq / min_d_sq
    
    def exact_objective_grad(params):
        """Compute exact objective and gradient using analytical formulas."""
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-10:
            return 1e10, np.zeros_like(params)
        
        obj = max_d_sq / min_d_sq
        
        # Gradient computation
        # d(obj)/d(d_sq_ij) for the min pair: -max_d_sq / min_d_sq^2
        # d(obj)/d(d_sq_ij) for the max pair: 1 / min_d_sq
        # d(d_sq_ij)/d(pts_k) = 2*(pts_i - pts_j) if k==i, -2*(pts_i - pts_j) if k==j, 0 otherwise
        
        min_idx = np.argmin(d_sq)
        max_idx = np.argmax(d_sq)
        
        # Get the pairs
        min_pair_i, min_pair_j = iu[0][min_idx], iu[1][min_idx]
        max_pair_i, max_pair_j = iu[0][max_idx], iu[1][max_idx]
        
        grad_pts = np.zeros_like(pts_opt)
        
        # Gradient contribution from min distance
        v_min = pts_opt[min_pair_i] - pts_opt[min_pair_j]
        grad_pts[min_pair_i] += 2 * v_min * (-max_d_sq / min_d_sq**2)
        grad_pts[min_pair_j] += -2 * v_min * (-max_d_sq / min_d_sq**2)
        
        # Gradient contribution from max distance
        v_max = pts_opt[max_pair_i] - pts_opt[max_pair_j]
        grad_pts[max_pair_i] += 2 * v_max * (1 / min_d_sq)
        grad_pts[max_pair_j] += -2 * v_max * (1 / min_d_sq)
        
        # Extract gradient for free parameters (indices 2+)
        grad_params = grad_pts[2:].flatten()
        
        return obj, grad_params
    
    # Smooth surrogate using log-sum-exp with adaptive beta
    def soft_objective(params, beta=50.0):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Soft max and min using log-sum-exp (numerically stable)
        max_d = np.max(d_sq)
        min_d = np.min(d_sq)
        
        # Stable log-sum-exp for max
        shifted_max = d_sq - max_d
        max_d_sq_soft = max_d + np.log(np.sum(np.exp(beta * shifted_max))) / beta
        
        # Stable log-sum-exp for min
        shifted_min = d_sq - min_d
        min_d_sq_soft = min_d - np.log(np.sum(np.exp(-beta * shifted_min))) / beta
        
        if min_d_sq_soft < 1e-10:
            return 1e10
        
        return max_d_sq_soft / min_d_sq_soft
    
    bnds = [(-15, 15)] * (2 * (N - 2))
    
    best_params = init_params
    best_score = exact_objective(init_params)
    
    # Phase 1: Smooth surrogate optimization with adaptive beta
    if time.time() < deadline - 40:
        try:
            # Start with lower beta, increase for better convergence
            for beta_val in [30.0, 50.0, 80.0]:
                if time.time() > deadline - 35:
                    break
                res = minimize(lambda p: soft_objective(p, beta=beta_val), init_params, 
                             method='L-BFGS-B', bounds=bnds,
                             options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < 1e9:
                    exact_score = exact_objective(res.x)
                    if exact_score < best_score:
                        best_params = res.x
                        best_score = exact_score
                        init_params = res.x  # Use improved solution as start for next beta
        except Exception:
            pass
    
    # Phase 2: Exact objective with analytical gradient
    if time.time() < deadline - 20:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           jac=exact_objective_grad,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Phase 3: High-precision polish with analytical gradient
    if time.time() < deadline - 10:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           jac=exact_objective_grad,
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Try multiple random restarts if time permits
    if time.time() < deadline - 15:
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb the best solution with varying scales
            scale = 0.05 + 0.05 * restart
            perturbation = rng.normal(0, scale, size=init_params.shape)
            restart_params = best_params + perturbation
            restart_score = exact_objective(restart_params)
            
            if restart_score < 1e9:
                try:
                    res = minimize(exact_objective, restart_params, method='L-BFGS-B', bounds=bnds,
                                   jac=exact_objective_grad,
                                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                    if res.fun < best_score:
                        best_params = res.x
                        best_score = res.fun
                except Exception:
                    pass
    
    # PHASE 4: Spectral regularization based optimization
    # Use Laplacian eigenvalues to encourage isotropic distribution
    if time.time() < deadline - 5:
        try:
            def spectral_objective(params):
                pts_opt = np.zeros((N, 2))
                pts_opt[0] = [0.0, 0.0]
                pts_opt[1] = [1.0, 0.0]
                pts_opt[2:] = params.reshape(-1, 2)
                
                diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                iu = np.triu_indices(N, 1)
                d_sq = dists_sq[iu]
                
                min_d_sq = np.min(d_sq)
                max_d_sq = np.max(d_sq)
                
                if min_d_sq < 1e-10:
                    return 1e10
                
                base_obj = max_d_sq / min_d_sq
                
                # Add spectral regularization: encourage uniform distribution
                # Compute graph Laplacian of complete graph with edge weights = 1/d_sq
                W = 1.0 / np.maximum(d_sq, 1e-10)
                W = np.where(np.eye(N, dtype=bool), 0, W)
                W = (W + W.T) / 2
                L = np.diag(W.sum(axis=1)) - W
                
                # Compute eigenvalues of Laplacian
                try:
                    eigvals = np.linalg.eigvalsh(L)
                    # Spectral gap is second smallest eigenvalue
                    # Larger gap = better connected = more uniform
                    spectral_gap = eigvals[1] if N > 1 else 0
                except Exception:
                    spectral_gap = 0
                
                # Regularization: penalize low spectral gap
                reg_weight = 0.01
                return base_obj + reg_weight / (1 + spectral_gap)
            
            res_spec = minimize(spectral_objective, best_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res_spec.fun < 1e9:
                exact_score = exact_objective(res_spec.x)
                if exact_score < best_score:
                    best_params = res_spec.x
                    best_score = exact_score
                    
        except Exception:
            pass
    
    # PHASE 5: Final high-precision polish with multiple strategies
    if time.time() < deadline - 2:
        try:
            # Try Nelder-Mead for local refinement
            res_nm = minimize(exact_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            if res_nm.fun < best_score:
                best_params = res_nm.x
                best_score = res_nm.fun
        except Exception:
            pass
        
        try:
            # Final L-BFGS-B polish
            res_final = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                                jac=exact_objective_grad,
                                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14})
            if res_final.fun < best_score:
                best_params = res_final.x
                best_score = res_final.fun
        except Exception:
            pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Final validation: ensure all points are distinct
    diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    min_dist = np.min(dists[iu])
    
    if min_dist < 1e-8:
        # Fallback: return hexagonal lattice subset
        R = 8
        pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                pts.append((x, y))
        pts = np.array(pts)
        dists_from_origin = np.sum(pts**2, axis=1)
        sorted_indices = np.argsort(dists_from_origin)
        return pts[sorted_indices[:N]]
    
    return final_pts

# EVOLVE-BLOCK-END
