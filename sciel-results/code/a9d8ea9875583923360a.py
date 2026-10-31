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
    
    # Compute all pairwise distances in the hex lattice
    # We want to find a subset of 16 points that minimizes (max_d / min_d)^2
    
    # For efficiency, let's use a greedy approach or local search.
    # First, let's generate a good initial guess.
    # The optimal configuration is likely a subset of the hex lattice.
    
    # Let's try to find the best subset of 16 points from a smaller region.
    # Sort points by distance from origin and take the first few candidates.
    
    # Actually, a better approach is to use local search on the lattice.
    # Start with the 16 closest points to the origin, then try swapping points.
    
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

    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N].tolist()
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)
    
    # Local search: try swapping one point in the subset with one outside
    # We only consider points within a reasonable range
    candidate_indices = sorted_indices[:100].tolist()  # Only consider top 100 candidates
    
    # Create a set for fast lookup
    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]
    
    # Iterate: try all swaps
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(N):
            if time.time() > deadline - 10:
                break
            for j in outside:
                if time.time() > deadline - 10:
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
    
    # Now, refine with continuous optimization.
    # Fix the first two points to remove translation and rotation invariance.
    # Find the two closest points in best_pts.
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    
    # Find the pair with minimum distance
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair[0], pair[1]
    if i0 > i1:
        i0, i1 = i1, i0
    
    p0 = best_pts[i0]
    p1 = best_pts[i1]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return best_pts
    
    # Rotation matrix
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform all points
    transformed = (best_pts - p0) @ R_mat.T / norm_v
    
    # Reorder so that i0 is at index 0 and i1 is at index 1
    # Create a permutation
    perm = list(range(N))
    # Move i0 to position 0
    perm[0], perm[i0] = perm[i0], perm[0]
    # Move i1 to position 1
    # Note: i1 might have been moved if i0 was 0 or 1
    # Let's do it more carefully
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    # Fill the rest
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    # Now transformed[0] is (0,0) and transformed[1] is (1,0)
    init_params = transformed[2:].flatten()
    
    # Objective function
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
    
    # Bounds for L-BFGS-B
    bnds = [(-10, 10)] * 28
    
    # Run L-BFGS-B
    try:
        res = minimize(objective, init_params, method='L-BFGS-B', bounds=bnds, 
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        
        best_params = res.x
        best_score = res.fun
        
    except Exception:
        best_params = init_params
        best_score = objective(init_params)
    
    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
