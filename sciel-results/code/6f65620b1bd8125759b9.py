"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate a hexagonal lattice
    R = 6
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    
    hex_pts = np.array(hex_pts)
    
    # 2. Helper functions
    def score_points(pts):
        """Calculate the exact score: (max dist / min dist)^2"""
        if len(pts) < 2:
            return 1e10
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        ratio_sq = max_d_sq / min_d_sq
        return ratio_sq

    def normalize_and_fix(pts):
        """
        Translate, rotate, and scale the points such that:
        - Point 0 is at (0,0)
        - Point 1 is at (1,0)
        Returns the parameters for the remaining N-2 points.
        """
        # Calculate all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
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
            return np.zeros((N-2)*2)
        
        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        # Transform all points: translate by -p0, rotate, scale by 1/norm_v
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        # Reorder so that i0 is at index 0 and i1 is at index 1
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # Now transformed[0] is (0,0) and transformed[1] is (1,0)
        # The parameters are the coordinates of points 2..N-1
        init_params = transformed[2:].flatten()
        return init_params

    def get_dists_from_params(params):
        """Reconstruct points from fixed points and params, return pairwise distances."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        return dists[iu]

    def smooth_objective(params, beta):
        """Smooth surrogate using log-sum-exp for max and min."""
        d = get_dists_from_params(params)
        if np.any(d < 1e-8):
            return 1e10
        
        # Soft Max: (1/beta) * log(sum(exp(beta * d)))
        # Numerically stable: max_d + (1/beta) * log(sum(exp(beta * (d - max_d))))
        max_d = np.max(d)
        min_d = np.min(d)
        
        # Soft Max
        max_d_soft = max_d + (1.0 / beta) * (logsumexp(beta * (d - max_d)))
        
        # Soft Min: min_d - (1/beta) * log(sum(exp(-beta * (d - min_d))))
        min_d_soft = min_d - (1.0 / beta) * (logsumexp(-beta * (d - min_d)))
        
        if min_d_soft < 1e-8:
            return 1e10
            
        ratio = max_d_soft / min_d_soft
        return ratio**2

    def exact_objective(params):
        """Exact objective for final evaluation."""
        d = get_dists_from_params(params)
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    # 3. Generate initial candidates
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    
    candidates = []
    
    # Candidate 1: Standard closest 16
    base_indices = sorted_indices[:N].tolist()
    candidates.append(hex_pts[base_indices])
    
    # Multi-start: Perturb the base configuration with various noise levels
    for i in range(10):
        pts = hex_pts[base_indices].copy()
        np.random.seed(i)
        noise_scale = 0.02 * (i + 1)
        noise = np.random.normal(0, noise_scale, pts.shape)
        pts += noise
        candidates.append(pts)
    
    # Try swapping boundary points
    for swap_idx in range(16, 22):
        if swap_idx < len(sorted_indices):
            new_indices = sorted_indices[:N].tolist()
            # Remove the farthest point and add the next closest
            farthest_in_subset = max(range(N), key=lambda k: dists_from_origin[sorted_indices[k]])
            new_indices[farthest_in_subset] = sorted_indices[swap_idx]
            pts = hex_pts[new_indices]
            candidates.append(pts)

    best_score = 1e10
    best_pts = None
    best_params = None

    bnds = [(-5, 5)] * (2 * (N - 2))

    for idx, init_pts in enumerate(candidates):
        if time.time() > deadline - 25:
            break
            
        # Normalize and get initial params
        init_params = normalize_and_fix(init_pts)
        
        # Initial score check
        current_score = exact_objective(init_params)
        
        # Optimization phases with smooth surrogates
        # Phase 1: Coarse, low beta
        if time.time() < deadline - 50:
            try:
                res1 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(10.0,),
                               options={'maxiter': 150, 'ftol': 1e-6, 'gtol': 1e-4})
                if res1.fun < 1e9:
                    candidate_params = res1.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass

        # Phase 2: Medium, higher beta
        if time.time() < deadline - 30:
            try:
                res2 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(50.0,),
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
                if res2.fun < 1e9:
                    candidate_params = res2.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass

        # Phase 3: Fine, very high beta
        if time.time() < deadline - 15:
            try:
                res3 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(200.0,),
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                if res3.fun < 1e9:
                    candidate_params = res3.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass
        
        # Phase 4: Exact polish with L-BFGS-B
        if time.time() < deadline - 8:
            try:
                res4 = minimize(exact_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                if res4.fun < 1e9:
                    candidate_params = res4.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass
        
        # Phase 5: Final high-precision polish with Nelder-Mead
        if time.time() < deadline - 3:
            try:
                res5 = minimize(exact_objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12})
                if res5.fun < 1e9:
                    candidate_params = res5.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass

        # Update global best
        if current_score < best_score:
            best_score = current_score
            best_params = init_params.copy()
            final_pts = np.zeros((N, 2))
            final_pts[0] = [0.0, 0.0]
            final_pts[1] = [1.0, 0.0]
            final_pts[2:] = best_params.reshape(-1, 2)
            best_pts = final_pts

    if best_pts is None:
        best_pts = hex_pts[base_indices]
        
    # Final verification
    final_score = score_points(best_pts)
    
    if len(best_pts) != N or best_pts.shape[1] != D:
        return np.zeros((N, D))
        
    return best_pts

# EVOLVE-BLOCK-END
