"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave margin for 150s limit

    def score_points(pts):
        """Compute (max_d / min_d)^2 for a set of points."""
        # Use squared distances for efficiency where possible, but we need actual ratio
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        # (max_d / min_d)^2 = (sqrt(max_d_sq) / sqrt(min_d_sq))^2 = max_d_sq / min_d_sq
        return max_d_sq / min_d_sq

    # 1. Generate initial candidates from various structured configurations
    candidates = []
    
    # Candidate 1: Hexagonal lattice subset (best known packing)
    # The optimal configuration for 16 points is likely related to hexagonal packing.
    # We generate a hex lattice and pick the 16 closest to the center.
    R = 6
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    candidates.append(hex_pts[sorted_indices[:N]])
    
    # Candidate 2: Grid subset
    G = 4
    grid_pts = []
    for i in range(-G, G+1):
        for j in range(-G, G+1):
            grid_pts.append((i, j))
    grid_pts = np.array(grid_pts)
    dists_from_origin = np.sum(grid_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    candidates.append(grid_pts[sorted_indices[:N]])
    
    # Candidate 3: Random initialization (Gaussian)
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.randn(N, 2)
        candidates.append(rand_pts)
        
    # Candidate 4: Spherical distribution (points on a circle + center?)
    # Actually, let's try a few random restarts with different seeds for robustness
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        rand_pts = np.random.randn(N, 2)
        candidates.append(rand_pts)

    # 2. Local Optimization using L-BFGS-B
    # We fix the first two points to remove translation and rotation invariance.
    # p0 = (0,0), p1 = (1,0).
    
    best_params = None
    best_score = float('inf')
    
    # Precompute indices for objective
    iu = np.triu_indices(N, 1)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        
        return max_d_sq / min_d_sq

    bnds = [(-10, 10)] * (2 * (N - 2))
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Normalize candidate to remove translation/rotation and scale
        # Find the two closest points in the candidate
        diff = cand[:, np.newaxis, :] - cand[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        d_flat = dists[iu]
        min_idx = np.argmin(d_flat)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = cand[i0]
        p1 = cand[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            continue
            
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (cand - p0) @ R_mat.T / norm_v
        
        # Reorder so i0 is at 0 and i1 is at 1
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        # Check if transformed[0] and transformed[1] are close to (0,0) and (1,0)
        # They should be by construction, but let's verify
        # init_params = transformed[2:].flatten()
        
        # Sometimes the local minimum is far, so we add a small perturbation
        # to help escape symmetry traps
        init_params = transformed[2:].flatten()
        
        # Run L-BFGS-B
        try:
            res = minimize(objective, init_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception as e:
            # If minimization fails, try a few manual steps or skip
            pass

    # If no valid result from optimization, fallback to best candidate
    if best_params is None:
        # Fallback: return the first candidate normalized
        cand = candidates[0]
        diff = cand[:, np.newaxis, :] - cand[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        d_flat = dists[iu]
        min_idx = np.argmin(d_flat)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0
        p0 = cand[i0]
        p1 = cand[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        transformed = (cand - p0) @ R_mat.T / norm_v
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = transformed[2:]
        return final_pts

    # 3. Refine with Nelder-Mead for better precision
    # L-BFGS-B can get stuck in flat regions or converge slowly near the optimum.
    # Nelder-Mead is derivative-free and can polish the solution.
    
    def objective_nm(params):
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        
        return max_d_sq / min_d_sq

    try:
        res_nm = minimize(objective_nm, best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_params = res_nm.x.copy()
            
    except Exception:
        pass

    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Verify distinctness
    diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    if np.min(d) < 1e-8:
        # Points are too close, add small perturbation
        # This shouldn't happen with the constraints, but just in case
        for i in range(N):
            for j in range(i):
                if np.linalg.norm(final_pts[i] - final_pts[j]) < 1e-8:
                    final_pts[i] += np.array([1e-6, 0])
                    
    return final_pts

# EVOLVE-BLOCK-END
