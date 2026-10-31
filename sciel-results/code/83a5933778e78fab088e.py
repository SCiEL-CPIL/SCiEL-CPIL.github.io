import numpy as np
import time
import itertools
from scipy.optimize import minimize, basinhopping

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    # P is (N, 2)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _soft_spread_objective(params, T=1.0):
    """
    Smooth surrogate objective.
    Minimizes the soft spread of squared distances.
    params: flattened array of shape (2*(N-2),) representing points 2..N-1.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    # Soft max and soft min using log-sum-exp
    m = np.max(d_sq)
    exp_max = np.exp(d_sq - m)
    soft_max = m + np.log(np.sum(exp_max))
    
    m_neg = np.max(-d_sq) # which is -min(d_sq)
    exp_min = np.exp(-d_sq + m_neg)
    soft_min = -m_neg - np.log(np.sum(exp_min))
    
    if soft_min < 1e-6:
        return 1e10
    
    # Objective: soft_max / soft_min
    return soft_max / soft_min

def _exact_objective(params):
    """
    Exact objective for local refinement.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _get_initial_candidates():
    """
    Generate diverse initial configurations.
    Returns list of (N, 2) arrays.
    """
    candidates = []
    
    # 1. Hexagonal Lattice Subset
    # Generate hex lattice points within radius sqrt(4)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    
    # Identify points by distance squared from origin
    d_sqs = np.sum(hex_points**2, axis=1)
    
    # We have 19 points. Need to remove 3.
    # Strategy: Remove 3 points that are "redundant" or on the boundary.
    # The outer ring has 6 points with d_sq=4.
    # The next inner ring has 6 points with d_sq=3.
    # The next has 6 points with d_sq=1.
    # Center is 1 point with d_sq=0.
    
    # Candidate A: Remove 3 from outer ring (d_sq=4)
    outer_idx = np.where(np.isclose(d_sqs, 4.0))[0]
    if len(outer_idx) >= 3:
        # Remove first 3
        keep_idx = [i for i in range(len(hex_points)) if i not in outer_idx[:3]]
        candidates.append(hex_points[keep_idx])
        
        # Remove different 3
        rng = np.random.default_rng(42)
        for _ in range(5):
            remove_idx = rng.choice(outer_idx, 3, replace=False)
            keep_idx = [i for i in range(len(hex_points)) if i not in remove_idx]
            candidates.append(hex_points[keep_idx])
            
    # Candidate B: Remove 2 from outer ring, 1 from inner ring (d_sq=3)
    inner_idx = np.where(np.isclose(d_sqs, 3.0))[0]
    if len(outer_idx) >= 2 and len(inner_idx) >= 1:
        remove_idx = list(outer_idx[:2]) + list(inner_idx[:1])
        keep_idx = [i for i in range(len(hex_points)) if i not in remove_idx]
        candidates.append(hex_points[keep_idx])
        
    # Candidate C: Remove 1 from outer ring, 2 from inner ring
    if len(outer_idx) >= 1 and len(inner_idx) >= 2:
        remove_idx = list(outer_idx[:1]) + list(inner_idx[:2])
        keep_idx = [i for i in range(len(hex_points)) if i not in remove_idx]
        candidates.append(hex_points[keep_idx])

    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 3. Two Concentric Circles
    pts_circ = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_circ.append([2 * np.cos(ang), 2 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_circ.append([1 * np.cos(ang), 1 * np.sin(ang)])
    candidates.append(np.array(pts_circ))
    
    # 4. Random Perturbed Grid
    rng = np.random.default_rng(42)
    perturbed = grid + rng.normal(0, 0.1, grid.shape)
    candidates.append(perturbed)
    
    # 5. Random Gaussian Cluster
    rng = np.random.default_rng(123)
    gauss = rng.normal(0, 1, (N, 2))
    candidates.append(gauss)
    
    # 6. Ring of 16
    pts_ring = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_ring.append([2 * np.cos(ang), 2 * np.sin(ang)])
    candidates.append(np.array(pts_ring))
    
    return candidates

def _prepare_for_optimization(P):
    """
    Fix P[0]=(0,0) and P[1]=(1,0) by finding the min-distance pair in P,
    rotating/ translating/scaling the whole set so that pair becomes (0,0) and (1,0).
    Returns the optimized parameter vector (coords of P[2:] to P[15]).
    """
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    p, q = np.unravel_index(min_idx, (N, N))
    if p > q:
        p, q = q, p
    
    p0 = P[p]
    p1 = P[q]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return None
        
    # Rotation matrix to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform all points
    transformed = (P - p0) @ R_mat.T / norm_v
    
    # Reorder points so that p is at index 0 and q is at index 1
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    # Parameters are the coordinates of points 2..15
    init_params = transformed[2:].flatten()
    
    return init_params

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    candidates = _get_initial_candidates()
    
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Evaluate initial candidates
    for P0 in candidates:
        init_params = _prepare_for_optimization(P0)
        if init_params is None:
            continue
        
        score = _exact_objective(init_params)
        if score < best_score:
            best_score = score
            best_params = init_params

    # Phase 2: Global Search with Basinhopping
    # Use the best initial candidate as the starting point for basinhopping
    if best_params is not None:
        # Define a callback to update best score
        def callback(x, f, accept):
            nonlocal best_score
            if f < best_score:
                best_score = f
                best_params = x.copy()
        
        # Run basinhopping
        try:
            bh_res = basinhopping(
                _soft_spread_objective,
                best_params,
                niter=20,  # Limit iterations to stay within time
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 50, 'ftol': 1e-8}},
                callback=callback,
                T=1.0,
                stepsize=0.5
            )
            if bh_res.fun < best_score:
                best_score = bh_res.fun
                best_params = bh_res.x
        except Exception:
            pass

    # Phase 3: Local Refinement
    if best_params is not None:
        # Refine with L-BFGS-B on soft objective
        try:
            res_soft = minimize(
                _soft_spread_objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_soft.fun < best_score:
                best_score = res_soft.fun
                best_params = res_soft.x
        except Exception:
            pass

        # Final polish with Nelder-Mead on exact objective
        try:
            res_final = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x
        except Exception:
            pass

    # Construct final points
    if best_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        # Check validity
        if _score(final_pts) > 1e6:
             # Fallback to best initial candidate
             return candidates[0]
             
        return final_pts
    else:
        # Fallback to first candidate
        return candidates[0]

# EVOLVE-BLOCK-END
