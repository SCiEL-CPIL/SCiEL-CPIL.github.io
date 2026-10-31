import numpy as np
import time
import itertools
from scipy.optimize import minimize

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
    # Reconstruct P
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    # Soft max and soft min using log-sum-exp
    # To avoid overflow, subtract max/min
    m = np.max(d_sq)
    # soft_max = m + log(sum(exp(d_sq - m)))
    # soft_min = -(-m_neg) ... let's use standard formula
    # soft_min(d) = - log(sum(exp(-d)))
    
    # Stability
    exp_max = np.exp(d_sq - m)
    soft_max = m + np.log(np.sum(exp_max))
    
    m_neg = np.max(-d_sq) # which is -min(d_sq)
    exp_min = np.exp(-d_sq + m_neg)
    soft_min = -m_neg - np.log(np.sum(exp_min))
    
    # We want to minimize the ratio, which is roughly proportional to the spread
    # If soft_min is very small, the ratio is huge.
    if soft_min < 1e-6:
        return 1e10
    
    # Objective: (soft_max - soft_min) / soft_min approximates ratio - 1
    # Or just soft_max - soft_min
    # Using the ratio form is better for scale invariance, but we fixed scale.
    # Let's minimize (soft_max / soft_min)
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
    
    # 1. Hexagonal Lattice Subset - evaluate all removal combinations
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    
    # Identify outer ring (dist^2 == 4)
    d_sqs = np.sum(hex_points**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    
    # Evaluate all combinations of 3 removals from outer ring
    if len(outer_indices) >= 3:
        best_hex_score = float('inf')
        best_hex_pts = None
        for removal_combo in itertools.combinations(outer_indices, 3):
            removal_set = set(removal_combo)
            keep_idx = [i for i in range(len(hex_points)) if i not in removal_set]
            pts_hex = hex_points[keep_idx]
            s = _score(pts_hex)
            if s < best_hex_score:
                best_hex_score = s
                best_hex_pts = pts_hex
        if best_hex_pts is not None:
            candidates.append(best_hex_pts)
    
    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 3. Two Concentric Circles (8 inner, 8 outer)
    pts_circ = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_circ.append([2 * np.cos(ang), 2 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_circ.append([1 * np.cos(ang), 1 * np.sin(ang)])
    candidates.append(np.array(pts_circ))
    
    # 4. Regular 16-gon
    pts_poly = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_poly.append([np.cos(ang), np.sin(ang)])
    candidates.append(np.array(pts_poly))
    
    # 5. Two-ring circle: 6 inner at r=0.6, 10 outer at r=1.4
    pts_two_ring = []
    for k in range(6):
        ang = 2 * np.pi * k / 6
        pts_two_ring.append([0.6 * np.cos(ang), 0.6 * np.sin(ang)])
    for k in range(10):
        ang = 2 * np.pi * (k + 0.5) / 10
        pts_two_ring.append([1.4 * np.cos(ang), 1.4 * np.sin(ang)])
    candidates.append(np.array(pts_two_ring))
    
    # 6. Random Perturbed Grid
    rng = np.random.default_rng(42)
    perturbed = grid + rng.normal(0, 0.1, grid.shape)
    candidates.append(perturbed)
    
    return candidates

def _prepare_for_optimization(P):
    """
    Fix P[0]=(0,0) and P[1]=(1,0) by finding the min-distance pair in P,
    rotating/ translating/scaling the whole set so that pair becomes (0,0) and (1,0).
    Returns the optimized parameter vector (coords of P[2:] to P[15]) and the permutation used.
    """
    # Find min distance pair
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
        # Degenerate, return original flattened (bad)
        return None, None
        
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
    
    return init_params, perm

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    candidates = _get_initial_candidates()
    
    best_score = float('inf')
    best_params = None
    best_perm = None
    best_initial_P = None
    
    # Phase 1: Smooth Optimization (L-BFGS-B)
    # Allocate time budget for each candidate
    num_candidates = len(candidates)
    # We have ~110s total. Let's spend 20s on each candidate for smooth opt?
    # L-BFGS-B is fast. Let's just run it with a maxiter limit.
    
    for i, P0 in enumerate(candidates):
        if time.time() > deadline - 30: # Leave 30s for final polish
            break
            
        init_params, perm = _prepare_for_optimization(P0)
        if init_params is None:
            continue
            
        # Run L-BFGS-B on soft spread
        try:
            res = minimize(
                _soft_spread_objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Evaluate exact score of the result
            P_opt = np.zeros((N, 2))
            P_opt[0] = [0.0, 0.0]
            P_opt[1] = [1.0, 0.0]
            P_opt[2:] = res.x.reshape(-1, 2)
            
            score = _score(P_opt)
            
            if score < best_score:
                best_score = score
                best_params = res.x
                best_perm = perm
                best_initial_P = P0 # Keep track of origin if needed, though we return optimized
                
        except Exception:
            pass

    # Phase 2: Local Refinement (Powell) on the best candidate
    if best_params is not None:
        if time.time() < deadline - 5:
            try:
                res_final = minimize(
                    _exact_objective,
                    best_params,
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': 1000, 'disp': False}
                )
                
                P_final = np.zeros((N, 2))
                P_final[0] = [0.0, 0.0]
                P_final[1] = [1.0, 0.0]
                P_final[2:] = res_final.x.reshape(-1, 2)
                
                final_score = _score(P_final)
                
                if final_score < best_score:
                    best_score = final_score
                    best_params = res_final.x
                    best_perm = perm
                    
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish for extra precision
        if best_params is not None and time.time() < deadline - 3:
            try:
                res_nm = minimize(
                    _exact_objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                
                P_nm = np.zeros((N, 2))
                P_nm[0] = [0.0, 0.0]
                P_nm[1] = [1.0, 0.0]
                P_nm[2:] = res_nm.x.reshape(-1, 2)
                
                nm_score = _score(P_nm)
                
                if nm_score < best_score:
                    best_score = nm_score
                    best_params = res_nm.x
                    
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
        # Fallback: find the best initial candidate
        best_init = None
        best_init_score = float('inf')
        for P0 in candidates:
            s = _score(P0)
            if s < best_init_score:
                best_init_score = s
                best_init = P0
        if best_init is not None:
            return best_init
        return candidates[0]

# EVOLVE-BLOCK-END
