"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, leave margin

    def score_points(pts):
        # pts is an array of shape (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def normalize_and_optimize(pts):
        """Normalize points (fix first two) and run continuous optimization."""
        if time.time() > deadline - 5:
            return pts, score_points(pts)
        
        # Find the pair with minimum distance
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
            return pts, score_points(pts)
        
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
        
        # Now transformed[0] is (0,0) and transformed[1] is (1,0)
        init_params = transformed[2:].flatten()
        
        # Objective function
        def objective(params):
            if time.time() > deadline:
                return 1e10
            pts_local = np.zeros((N, 2))
            pts_local[0] = [0.0, 0.0]
            pts_local[1] = [1.0, 0.0]
            pts_local[2:] = params.reshape(-1, 2)
            
            diff_local = pts_local[:, np.newaxis, :] - pts_local[np.newaxis, :, :]
            dists_local = np.sqrt(np.sum(diff_local**2, axis=2))
            iu_local = np.triu_indices(N, 1)
            d_local = dists_local[iu_local]
            
            min_d_local = np.min(d_local)
            max_d_local = np.max(d_local)
            
            if min_d_local < 1e-8:
                return 1e10
            
            return (max_d_local / min_d_local)**2
        
        # Bounds for L-BFGS-B
        bnds = [(-5, 5)] * 28
        
        # Run L-BFGS-B with multiple attempts
        best_params = init_params
        best_score = objective(init_params)
        
        try:
            res = minimize(objective, init_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
        
        # Construct final points
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        return final_pts, best_score

    # Strategy: Try multiple initial configurations and optimize each
    best_pts = None
    best_score = 1e10

    # 1. Hexagonal lattice subset
    R = 4
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    
    # Try multiple subsets from hex lattice
    for start_idx in range(0, 30, 5):
        if time.time() > deadline - 20:
            break
        subset = sorted_indices[start_idx:start_idx + N]
        pts = hex_pts[subset]
        # Local search on lattice
        candidate_indices = sorted_indices[:60]
        in_subset = set(subset)
        outside = [i for i in candidate_indices if i not in in_subset]
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for i in range(N):
                if time.time() > deadline - 15:
                    break
                for j in outside:
                    if time.time() > deadline - 15:
                        break
                    new_subset = subset.copy()
                    new_subset[i] = j
                    new_pts = hex_pts[new_subset]
                    new_score = score_points(new_pts)
                    if new_score < best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                        subset = new_subset
                        in_subset = set(subset)
                        outside = [k for k in candidate_indices if k not in in_subset]
                        improved = True
        
        # Optimize the best lattice configuration found
        if best_pts is not None:
            opt_pts, opt_score = normalize_and_optimize(best_pts)
            if opt_score < best_score:
                best_score = opt_score
                best_pts = opt_pts

    # 2. Random restarts
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        # Random points in a disk
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Optimize
        opt_pts, opt_score = normalize_and_optimize(pts)
        if opt_score < best_score:
            best_score = opt_score
            best_pts = opt_pts

    # 3. Square lattice subset
    S = 4
    sq_pts = []
    for i in range(-S, S + 1):
        for j in range(-S, S + 1):
            sq_pts.append((float(i), float(j)))
    sq_pts = np.array(sq_pts)
    
    dists_sq = np.sum(sq_pts**2, axis=1)
    sorted_sq = np.argsort(dists_sq)
    
    for start_idx in range(0, 20, 5):
        if time.time() > deadline - 5:
            break
        subset = sorted_sq[start_idx:start_idx + N]
        pts = sq_pts[subset]
        opt_pts, opt_score = normalize_and_optimize(pts)
        if opt_score < best_score:
            best_score = opt_score
            best_pts = opt_pts

    if best_pts is None:
        # Fallback: simple grid
        best_pts = np.array([[i % 4, i // 4] for i in range(N)], dtype=float)

    return best_pts

# EVOLVE-BLOCK-END
