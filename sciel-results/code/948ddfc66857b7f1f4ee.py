"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, leave margin

    def score_points(pts):
        """Compute (max_dist / min_dist)^2 for a set of points."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    # Generate a hexagonal lattice with enough points
    R = 6
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    
    hex_pts = np.array(hex_pts)
    n_hex = len(hex_pts)
    
    # Compute all pairwise distances in the hex lattice for efficiency
    diff = hex_pts[:, np.newaxis, :] - hex_pts[np.newaxis, :, :]
    dists_all = np.sqrt(np.sum(diff**2, axis=2))
    iu_all = np.triu_indices(n_hex, 1)
    dists_flat = dists_all[iu_all]
    
    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N].tolist()
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)
    
    # Also try the hexagonal cluster approach from current program
    sqrt3 = np.sqrt(3)
    def hex_points(radius):
        pts = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                if max(abs(q), abs(r), abs(q + r)) <= radius:
                    x = q + 0.5 * r
                    y = (sqrt3 / 2) * r
                    pts.append((x, y))
        return np.array(pts)
    
    # Try radius 2 hexagon (19 points) - remove 3 to get 16
    hex2 = hex_points(2)
    n_hex2 = len(hex2)
    
    if n_hex2 >= N:
        indices = list(range(n_hex2))
        for combo in combinations(indices, N):
            if time.time() > deadline - 30:
                break
            pts = hex2[list(combo)]
            s = score_points(pts)
            if s < best_score:
                best_score = s
                best_pts = pts.copy()
    
    # Local search on the lattice: try swapping one point in the subset with one outside
    # We use the precomputed distances for efficiency
    in_subset = set(initial_subset)
    candidate_indices = sorted_indices[:150].tolist()
    outside = [i for i in candidate_indices if i not in in_subset]
    
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for i in range(N):
            if time.time() > deadline - 20:
                break
            for j in outside:
                if time.time() > deadline - 20:
                    break
                # Swap i and j
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
    
    # Continuous optimization refinement
    if time.time() < deadline - 5:
        # Fix first two points to remove translation/rotation invariance
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = best_pts[i0]
        p1 = best_pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v > 1e-8:
            cos_t = v[0] / norm_v
            sin_t = v[1] / norm_v
            R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
            
            transformed = (best_pts - p0) @ R_mat.T / norm_v
            
            # Reorder
            perm = list(range(N))
            perm[0] = i0
            perm[1] = i1
            remaining = [i for i in range(N) if i != i0 and i != i1]
            perm[2:] = remaining
            transformed = transformed[perm]
            
            init_params = transformed[2:].flatten()
            
            def objective(params):
                if time.time() > deadline:
                    return 1e10
                pts = np.zeros((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = params.reshape(-1, 2)
                
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                iu = np.triu_indices(N, 1)
                d = dists[iu]
                
                min_d = np.min(d)
                max_d = np.max(d)
                
                if min_d < 1e-8:
                    return 1e10
                
                return (max_d / min_d)**2
            
            bnds = [(-10, 10)] * 28
            
            try:
                res = minimize(objective, init_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if res.fun < best_score:
                    best_score = res.fun
                    final_pts = np.zeros((N, 2))
                    final_pts[0] = [0.0, 0.0]
                    final_pts[1] = [1.0, 0.0]
                    final_pts[2:] = res.x.reshape(-1, 2)
                    best_pts = final_pts
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: return a simple configuration
        best_pts = np.zeros((N, 2))
        for i in range(N):
            best_pts[i] = [i, 0]
    
    return best_pts

# EVOLVE-BLOCK-END
