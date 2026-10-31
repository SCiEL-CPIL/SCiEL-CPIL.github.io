import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def objective(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        Using squared distances: (max_d / min_d)^2 = (max_d_sq / min_d_sq)
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Check for coincident points
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
            
        max_d_sq = np.max(d_sq)
        
        return max_d_sq / min_d_sq

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
            # Degenerate case, return zeros or handle error
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
        # Soft max: (1/beta) * log(sum(exp(beta * d)))
        # To avoid overflow, use max_d as reference
        max_d = np.max(d)
        min_d = np.min(d)
        
        # Numerically stable log-sum-exp
        # max_d_soft = max_d + (1/beta) * log(sum(exp(beta * (d - max_d))))
        # min_d_soft = min_d - (1/beta) * log(sum(exp(-beta * (d - min_d))))
        
        # However, we want to minimize (max/min)^2.
        # Let's compute soft max and soft min.
        
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
        return ratio**2

    def exact_objective(params):
        """Exact objective for final evaluation."""
        d = get_dists_from_params(params)
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def make_hexagonal_cluster(n):
        """
        Generate a hexagonal lattice cluster of n points centered at origin.
        """
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        
        # Generate a large enough grid of lattice points
        pts = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        
        # Center the cluster
        pts -= pts.mean(axis=0)
        
        # Sort by distance to center and pick the closest n
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        return pts[order[:n]]

    # Initial best candidate from hexagonal cluster
    init_pts = make_hexagonal_cluster(N)
    best_val = objective(init_pts)
    best_pts = init_pts.copy()
    
    # Generate additional diverse initial candidates
    candidates = [init_pts]
    
    # Rotated hexagonal clusters
    for angle in [10, 20, 30, 45, 60, 90]:
        rad = np.deg2rad(angle)
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        candidates.append(init_pts @ rot.T)
        
    # Perturbed hexagonal clusters
    np.random.seed(42)
    for _ in range(3):
        noise = np.random.normal(0, 0.1, init_pts.shape)
        candidates.append(init_pts + noise)
        
    # 4x4 grid
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid_pts = np.column_stack([x.ravel(), y.ravel()])
    candidates.append(grid_pts)
    
    # Circular distribution
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    circle1 = np.column_stack([np.cos(angles), np.sin(angles)])
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    circle2 = np.column_stack([np.cos(angles2), np.sin(angles2)]) * 0.5
    candidates.append(np.vstack([circle1, circle2]))

    bnds = [(-5, 5)] * (2 * (N - 2))
    
    # Optimize each candidate using the smooth objective approach
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break
            
        # Normalize and get initial params
        init_params = normalize_and_fix(cand)
        
        # Initial score check
        current_score = exact_objective(init_params)
        
        # Optimization phases
        # Phase 1: Coarse, low beta
        if time.time() < deadline - 50:
            try:
                res1 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(10.0,),
                               options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4})
                if res1.fun < 1e9:
                    candidate_params = res1.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass

        # Phase 2: Medium, higher beta
        if time.time() < deadline - 20:
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
        if time.time() < deadline - 10:
            try:
                res3 = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                               bounds=bnds, args=(100.0,),
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                if res3.fun < 1e9:
                    candidate_params = res3.x
                    cand_score = exact_objective(candidate_params)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = candidate_params
            except Exception:
                pass
        
        # Final exact polish
        if time.time() < deadline - 5:
            try:
                res4 = minimize(exact_objective, init_params, method='Powell',
                               options={'maxiter': 2000, 'maxfev': 20000})
                if res4.fun < current_score:
                    current_score = res4.fun
                    init_params = res4.x
            except Exception:
                pass
        
        # Reconstruct points and check if better
        if current_score < best_val:
            best_val = current_score
            pts = np.zeros((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = init_params.reshape(-1, 2)
            best_pts = pts.copy()

    # Run Differential Evolution on the best candidate found
    if time.time() < deadline - 30:
        try:
            bounds = [(-5.0, 5.0)] * (N * D)
            de_result = differential_evolution(
                objective, 
                bounds, 
                maxiter=200, 
                popsize=15, 
                tol=1e-8,
                seed=42,
                polish=True
            )
            
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_pts = de_result.x.reshape(N, D).copy()
        except Exception:
            pass

    # Final local polish using Powell on the best candidate found
    if time.time() < deadline - 10:
        try:
            current = best_pts.copy()
            result = minimize(
                objective,
                current,
                method='Powell',
                options={'maxiter': 5000, 'maxfev': 50000}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_pts = result.x.copy()
        except Exception:
            pass

    # Ensure we return valid distinct points
    return best_pts
