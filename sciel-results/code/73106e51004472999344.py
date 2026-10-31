"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave margin

    # Generate a hexagonal lattice with enough points
    R = 5
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    
    hex_pts = np.array(hex_pts)
    
    def score_points(pts):
        """Exact objective: (max_d / min_d)^2"""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def smooth_objective(params, beta=50.0):
        """Smooth surrogate using log-sum-exp for soft min/max."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Soft min of distances (using log-sum-exp on negative)
        # soft_min(d) = -1/beta * log(sum(exp(-beta * d)))
        # But we need soft_min of d, not d^2. Use d = sqrt(d_sq)
        d = np.sqrt(d_sq)
        
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        max_neg_beta_d = np.max(-beta * d)
        soft_min_d = -(max_neg_beta_d + np.log(np.sum(np.exp(-beta * d - max_neg_beta_d)))) / beta
        
        # Soft max: 1/beta * log(sum(exp(beta * d)))
        max_beta_d = np.max(beta * d)
        soft_max_d = (max_beta_d + np.log(np.sum(np.exp(beta * d - max_beta_d)))) / beta
        
        if soft_min_d < 1e-8:
            return 1e10
        
        return (soft_max_d / soft_min_d)**2

    def normalize_pts(pts):
        """Normalize points: translate so first point is at origin, rotate so second is on x-axis, scale so min dist = 1."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        # Find the pair with minimum distance
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return pts, None
        
        # Rotation matrix
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        # Transform all points
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        # Reorder so that i0 is at index 0 and i1 is at index 1
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        return transformed, perm

    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N].tolist()
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)
    
    # Local search: try swapping one point in the subset with one outside
    candidate_indices = sorted_indices[:100].tolist()
    
    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]
    
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i in range(N):
            if time.time() > deadline - 30:
                break
            for j in outside:
                if time.time() > deadline - 30:
                    break
                new_subset = initial_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = score_points(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    initial_subset = new_subset
                    in_subset = set(initial_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True
    
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)
    
    # Normalize for optimization
    norm_pts, _ = normalize_pts(best_pts)
    if norm_pts is None:
        return best_pts
    
    init_params = norm_pts[2:].flatten()
    
    # Bounds for L-BFGS-B
    bnds = [(-10, 10)] * 28
    
    best_params = init_params.copy()
    best_opt_score = best_score
    
    # Phase 1: Smooth surrogate optimization with multiple beta values
    for beta in [20.0, 50.0, 100.0]:
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(smooth_objective, best_params, args=(beta,), 
                          method='L-BFGS-B', bounds=bnds,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_opt_score:
                # Check with exact objective
                pts = np.zeros((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = res.x.reshape(-1, 2)
                exact_score = score_points(pts)
                if exact_score < best_opt_score:
                    best_opt_score = exact_score
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Exact objective optimization (polish)
    def exact_objective(params):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        return score_points(pts)
    
    try:
        res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
        
        if res.fun < best_opt_score:
            best_opt_score = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    # Phase 3: Try Nelder-Mead for local refinement
    try:
        res = minimize(exact_objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        
        if res.fun < best_opt_score:
            best_opt_score = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    # Final polish with tighter tolerances
    try:
        res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                       options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14})
        
        if res.fun < best_opt_score:
            best_opt_score = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    final_score = score_points(final_pts)
    
    # If optimization made things worse, fall back to lattice result
    if final_score > best_score:
        return best_pts
    
    return final_pts

# EVOLVE-BLOCK-END
