import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)

    def raw_score(pts):
        """
        Calculate the raw squared ratio of max to min pairwise distance.
        This is the actual metric we want to minimize.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        max_d_sq = np.max(d_sq)
        return max_d_sq / min_d_sq

    def anisotropy_penalty(pts):
        """
        Calculate a penalty for anisotropy (non-circular distribution).
        Uses the ratio of the maximum to minimum eigenvalue of the covariance matrix.
        Only used in early optimization phases to guide search.
        """
        if len(pts) < 2:
            return 0.0
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        if cov.shape != (2, 2):
            return 0.0
        eigenvalues = np.linalg.eigvalsh(cov)
        min_eig = np.clip(eigenvalues[0], 1e-10, None)
        max_eig = np.clip(eigenvalues[1], 1e-10, None)
        return max_eig / min_eig

    def normalize_and_fix(pts):
        """
        Translate, rotate, and scale the points such that:
        - The closest pair is at (0,0) and (1,0)
        Returns the parameters for the remaining N-2 points.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_idx = np.argmin(d_sq)
        # Map back to original indices
        # iu[0] and iu[1] are the row and column indices in the upper triangle
        i0 = iu[0][min_idx]
        i1 = iu[1][min_idx]
        
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
        
        init_params = transformed[2:].flatten()
        return init_params

    def get_pts_from_params(params):
        """Reconstruct points from fixed points and params."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        return pts

    def get_dists_from_params(params):
        """Reconstruct points from fixed points and params, return pairwise distances."""
        pts = get_pts_from_params(params)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        return dists[iu]

    def smooth_objective(params, beta, penalty_weight=0.0):
        """Smooth surrogate using log-sum-exp for max and min."""
        pts = get_pts_from_params(params)
        d = get_dists_from_params(params)
        if np.any(d < 1e-8):
            return 1e10
        max_d = np.max(d)
        min_d = np.min(d)
        
        # Soft Max
        d_shifted_max = d - max_d
        exp_terms_max = np.exp(beta * d_shifted_max)
        sum_exp_max = np.sum(exp_terms_max)
        if sum_exp_max < 1e-300:
            return 1e10
        max_d_soft = max_d + (1.0 / beta) * np.log(sum_exp_max)
        
        # Soft Min
        d_shifted_min = d - min_d
        exp_terms_min = np.exp(-beta * d_shifted_min)
        sum_exp_min = np.sum(exp_terms_min)
        if sum_exp_min < 1e-300:
            return 1e10
        min_d_soft = min_d - (1.0 / beta) * np.log(sum_exp_min)
        
        if min_d_soft < 1e-8:
            return 1e10
            
        ratio = max_d_soft / min_d_soft
        score = ratio**2
        
        if penalty_weight > 0:
            pen = anisotropy_penalty(pts)
            score += penalty_weight * pen
        return score

    def exact_objective(params, penalty_weight=0.0):
        """Exact objective."""
        pts = get_pts_from_params(params)
        d = get_dists_from_params(params)
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        score = (max_d / min_d)**2
        if penalty_weight > 0:
            pen = anisotropy_penalty(pts)
            score += penalty_weight * pen
        return score

    def make_hexagonal_cluster(n):
        """
        Generate a hexagonal lattice cluster of n points centered at origin.
        """
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        
        pts = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        pts -= pts.mean(axis=0)
        
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        return pts[order[:n]]

    # Generate initial candidates
    np.random.seed(42)
    
    # Hexagonal cluster
    hex_pts = make_hexagonal_cluster(N)
    candidates = [hex_pts.copy()]
    
    # Rotated hexagonal clusters
    for angle in [15, 30, 45, 60]:
        rad = np.deg2rad(angle)
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        candidates.append(hex_pts @ rot.T)
    
    # Perturbed hexagonal
    for sigma in [0.05, 0.1, 0.15]:
        noise = np.random.normal(0, sigma, hex_pts.shape)
        candidates.append(hex_pts + noise)
    
    # 4x4 grid (slightly perturbed to avoid degeneracy)
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid_pts = np.column_stack([x.ravel(), y.ravel()])
    candidates.append(grid_pts + np.random.normal(0, 0.01, grid_pts.shape))
    
    # Two concentric circles (8 + 8)
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    circle1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    circle2 = np.column_stack([np.cos(angles2), np.sin(angles2)]) * 0.6
    candidates.append(np.vstack([circle1, circle2]))
    
    # Three rings: 4 + 4 + 8
    a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    ring1 = np.column_stack([np.cos(a1), np.sin(a1)]) * 0.3
    a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    ring2 = np.column_stack([np.cos(a2), np.sin(a2)]) * 0.7
    a3 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ring3 = np.column_stack([np.cos(a3), np.sin(a3)]) * 1.1
    candidates.append(np.vstack([ring1, ring2, ring3]))
    
    # Random points in a disk
    for _ in range(3):
        r = np.sqrt(np.random.uniform(0, 1, N))
        theta = np.random.uniform(0, 2*np.pi, N)
        rand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(rand_pts)

    bnds = [(-5, 5)] * (2 * (N - 2))
    
    # Optimize each candidate
    best_val = 1e10
    best_pts = hex_pts.copy()
    
    # Evaluate all candidates first
    for cand in candidates:
        s = raw_score(cand)
        if s < best_val:
            best_val = s
            best_pts = cand.copy()
    
    # Time budget allocation
    # Phase 1: Smooth optimization with anisotropy penalty (guiding) - 60s
    # Phase 2: Smooth optimization without penalty - 30s  
    # Phase 3: Exact local optimization - 40s
    # Phase 4: Final polish - 15s
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
        
        init_params = normalize_and_fix(cand)
        current_score = raw_score(get_pts_from_params(init_params))
        
        # Phase 1: Smooth with penalty (guiding)
        if time.time() < deadline - 80:
            try:
                res1 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(5.0, 0.05),
                               options={'maxiter': 80, 'ftol': 1e-5, 'gtol': 1e-3})
                if res1.fun < 1e9:
                    cand_score = raw_score(get_pts_from_params(res1.x))
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = res1.x
            except Exception:
                pass

        # Phase 2: Smooth without penalty
        if time.time() < deadline - 50:
            try:
                res2 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(20.0, 0.0),
                               options={'maxiter': 150, 'ftol': 1e-7, 'gtol': 1e-5})
                if res2.fun < 1e9:
                    cand_score = raw_score(get_pts_from_params(res2.x))
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = res2.x
            except Exception:
                pass

        # Phase 3: Higher beta smooth
        if time.time() < deadline - 25:
            try:
                res3 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(100.0, 0.0),
                               options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7})
                if res3.fun < 1e9:
                    cand_score = raw_score(get_pts_from_params(res3.x))
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = res3.x
            except Exception:
                pass
        
        # Phase 4: Exact local polish
        if time.time() < deadline - 10:
            try:
                res4 = minimize(exact_objective, init_params, method='Powell',
                               options={'maxiter': 1500, 'maxfev': 15000})
                cand_score = raw_score(get_pts_from_params(res4.x))
                if cand_score < current_score:
                    current_score = cand_score
                    init_params = res4.x
            except Exception:
                pass
        
        if current_score < best_val:
            best_val = current_score
            best_pts = get_pts_from_params(init_params).copy()

    # Multiple local restarts from the best solution
    np.random.seed(43)
    for _ in range(15):
        if time.time() > deadline - 15:
            break
        
        noise = np.random.normal(0, 0.08, best_pts.shape)
        perturbed_pts = best_pts + noise
        
        init_params = normalize_and_fix(perturbed_pts)
        
        try:
            # Quick smooth pass
            res_s = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                           bounds=bnds, args=(50.0, 0.0),
                           options={'maxiter': 100, 'ftol': 1e-7, 'gtol': 1e-5})
            if res_s.fun < 1e9:
                init_params = res_s.x
            
            # Exact polish
            res = minimize(exact_objective, init_params, method='Powell',
                           options={'maxiter': 800, 'maxfev': 8000})
            cand_score = raw_score(get_pts_from_params(res.x))
            if cand_score < best_val:
                best_val = cand_score
                best_pts = get_pts_from_params(res.x).copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 3:
        try:
            init_params = normalize_and_fix(best_pts)
            
            # Very high beta smooth
            res_high = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(500.0, 0.0),
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_high.fun < 1e9:
                init_params = res_high.x
            
            # Exact Powell with tight tolerance
            res_final = minimize(exact_objective, init_params, method='Powell',
                               options={'maxiter': 5000, 'maxfev': 50000})
            
            final_score = raw_score(get_pts_from_params(res_final.x))
            if final_score < best_val:
                best_val = final_score
                best_pts = get_pts_from_params(res_final.x).copy()
        except Exception:
            pass

    return best_pts
