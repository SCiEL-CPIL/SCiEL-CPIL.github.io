"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, leave margin for safety

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

    def soft_ratio(pts, beta=50.0):
        """Soft surrogate for (max_d/min_d)^2 using log-sum-exp."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d2 = sq_dists[iu]
        
        # Soft max and min via log-sum-exp
        d2_clipped = np.clip(d2, 1e-10, 1e10)
        log_sum_exp_max = np.log(np.sum(np.exp(beta * d2_clipped))) / beta
        log_sum_exp_min = -np.log(np.sum(np.exp(-beta * d2_clipped))) / beta
        
        if log_sum_exp_min < 1e-8:
            return 1e10
        return (log_sum_exp_max / log_sum_exp_min)**2

    # Generate a hexagonal lattice with enough points
    R = 5
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    
    hex_pts = np.array(hex_pts)
    
    # Compute pairwise squared distances for all lattice points (for fast local search)
    # But that's O(M^2) where M is lattice size ~ 121, so 14641 pairs - manageable
    # Actually let's just compute on the fly but vectorize the local search
    
    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    
    # Try multiple initial subsets
    initial_candidates = []
    
    # Candidate 1: 16 closest to origin
    initial_candidates.append(sorted_indices[:N].tolist())
    
    # Candidate 2: 16 points with best local density (greedy)
    # Start with closest, then add points that maintain good min distance
    greedy_subset = [sorted_indices[0]]
    for idx in sorted_indices[1:]:
        if len(greedy_subset) >= N:
            break
        # Check if adding this point maintains reasonable min distance
        new_pts = hex_pts[greedy_subset + [idx]]
        diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(len(new_pts), 1)
        d = dists[iu]
        if np.min(d) > 0.5:  # Threshold for minimum distance
            greedy_subset.append(idx)
    
    # If greedy didn't get 16, pad with closest
    while len(greedy_subset) < N:
        for idx in sorted_indices:
            if idx not in greedy_subset:
                greedy_subset.append(idx)
                break
    initial_candidates.append(greedy_subset[:N])
    
    # Candidate 3: Ring-based selection
    ring_subset = []
    # Take points at various radial distances
    for radius_sq in [0, 1, 3, 4, 7, 9, 12, 13, 16, 19]:
        close = [i for i in range(len(hex_pts)) if abs(dists_from_origin[i] - radius_sq) < 0.5]
        for i in close:
            if i not in ring_subset and len(ring_subset) < N:
                ring_subset.append(i)
    while len(ring_subset) < N:
        for idx in sorted_indices:
            if idx not in ring_subset:
                ring_subset.append(idx)
                break
    initial_candidates.append(ring_subset[:N])
    
    # Evaluate all candidates
    best_score = float('inf')
    best_subset = None
    for cand in initial_candidates:
        s = score_points(hex_pts[cand])
        if s < best_score:
            best_score = s
            best_subset = cand
    
    # Local search: try swapping one point in the subset with one outside
    candidate_indices = sorted_indices[:80].tolist()  # Only consider top 80 candidates
    
    in_subset = set(best_subset)
    outside = [i for i in candidate_indices if i not in in_subset]
    
    # Iterate: try all swaps with early termination
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() > deadline - 30:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 30:
                break
            for j in outside:
                if time.time() > deadline - 30:
                    break
                # Swap i and j
                new_subset = best_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = score_points(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_subset = new_subset
                    in_subset = set(best_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True
        if not improved:
            break
    
    best_pts = hex_pts[best_subset]
    best_score = score_points(best_pts)
    
    # Now, refine with continuous optimization using multiple starts
    # Fix the first two points to remove translation and rotation invariance.
    
    def prepare_for_optimization(pts):
        """Prepare points for optimization by fixing two points."""
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
            return None
        
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
        
        return init_params
    
    # Objective function with soft surrogate option
    def objective(params, use_soft=False):
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        if use_soft:
            return soft_ratio(pts, beta=30.0)
        else:
            return score_points(pts)
    
    # Bounds for L-BFGS-B
    bnds = [(-10, 10)] * 28
    
    # Generate multiple initial configurations for optimization
    opt_inits = []
    
    # Init 1: From lattice solution
    init1 = prepare_for_optimization(best_pts)
    if init1 is not None:
        opt_inits.append(init1)
    
    # Init 2: Slightly perturbed lattice solution
    if init1 is not None:
        np.random.seed(42)
        init2 = init1 + np.random.normal(0, 0.1, size=init1.shape)
        opt_inits.append(init2)
    
    # Init 3: Regular grid-based initialization
    # 4x4 grid
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([i, j])
    grid_pts = np.array(grid_pts, dtype=float)
    # Normalize: scale so min distance is 1
    diff = grid_pts[:, np.newaxis, :] - grid_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(16, 1)
    d = dists[iu]
    min_d = np.min(d)
    if min_d > 1e-8:
        grid_pts = grid_pts / min_d
        # Center
        grid_pts = grid_pts - np.mean(grid_pts, axis=0)
        # Fix first two points
        init3 = prepare_for_optimization(grid_pts)
        if init3 is not None:
            opt_inits.append(init3)
    
    # Init 4: Hexagonal cluster
    hex_cluster = []
    # Central point and neighbors
    hex_cluster.append([0, 0])
    for angle in range(0, 360, 60):
        rad = np.radians(angle)
        hex_cluster.append([np.cos(rad), np.sin(rad)])
    # Second ring
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        hex_cluster.append([2*np.cos(rad), 2*np.sin(rad)])
    hex_cluster = np.array(hex_cluster[:16], dtype=float)
    # Normalize
    diff = hex_cluster[:, np.newaxis, :] - hex_cluster[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(16, 1)
    d = dists[iu]
    min_d = np.min(d)
    if min_d > 1e-8:
        hex_cluster = hex_cluster / min_d
        hex_cluster = hex_cluster - np.mean(hex_cluster, axis=0)
        init4 = prepare_for_optimization(hex_cluster)
        if init4 is not None:
            opt_inits.append(init4)
    
    # Run optimization from each initial point
    best_opt_params = None
    best_opt_score = float('inf')
    
    for init_params in opt_inits:
        if time.time() > deadline - 20:
            break
        
        # First try with soft surrogate
        try:
            res_soft = minimize(lambda p: objective(p, use_soft=True), init_params, 
                               method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res_soft.fun < best_opt_score:
                best_opt_score = res_soft.fun
                best_opt_params = res_soft.x.copy()
        except Exception:
            pass
        
        # Then refine with exact objective
        if time.time() > deadline - 15:
            break
        
        try:
            res_exact = minimize(lambda p: objective(p, use_soft=False), init_params,
                                method='L-BFGS-B', bounds=bnds,
                                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res_exact.fun < best_opt_score:
                best_opt_score = res_exact.fun
                best_opt_params = res_exact.x.copy()
        except Exception:
            pass
        
        # If we have a good result, do a final polish from the best found
        if best_opt_params is not None and best_opt_score < 1.45:
            if time.time() > deadline - 10:
                break
            try:
                res_polish = minimize(lambda p: objective(p, use_soft=False), best_opt_params,
                                     method='L-BFGS-B', bounds=bnds,
                                     options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
                
                if res_polish.fun < best_opt_score:
                    best_opt_score = res_polish.fun
                    best_opt_params = res_polish.x.copy()
            except Exception:
                pass
    
    # Construct final points
    if best_opt_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_opt_params.reshape(-1, 2)
        
        # Verify the score
        final_score = score_points(final_pts)
        
        # Use the better of optimization result and lattice result
        if final_score < best_score:
            return final_pts
        else:
            return best_pts
    else:
        return best_pts

# EVOLVE-BLOCK-END
