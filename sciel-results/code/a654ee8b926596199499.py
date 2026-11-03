"""Pack 50 equal circles in the unit disk via ring-based parametric optimization + gradient polish + binary search radius."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0  # safety margin before 150s limit


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -np.inf
    n = len(pts)
    # Container constraint
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_container, min_pair)


def _energy(P, r):
    """
    Smooth energy function: sum of squared overlaps.
    """
    # Circle-circle overlaps
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    
    # Circle-boundary overlaps
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))


def _energy_grad(P, r):
    """
    Gradient of the energy function.
    """
    grad = np.zeros_like(P)
    
    # Pairwise
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    V_cc = np.maximum(2.0 * r - dist, 0.0)
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    W = V_cc * inv_dist
    
    sum_W = np.sum(W, axis=1) 
    term1 = (P.T @ W).T 
    term2 = P * sum_W[:, np.newaxis] 
    
    grad += term1 - term2
    
    # Boundary
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r + dist_origin - 1.0, 0.0)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad


def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    
    # Radii are the first k params
    radii = params[:k]
    # Offsets are the next k params
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            # Center point
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
            
    return P


def _optimize_positions(P_init, r_target, maxiter=300):
    """
    Optimize positions P to minimize energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())


def _global_search_ring_params(ring_counts, seed=42, maxiter=80):
    """
    Use differential evolution to find good ring radii and offsets.
    """
    k = len(ring_counts)
    # Bounds for radii: 0 to 1
    # Bounds for offsets: 0 to 2*pi
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        P = _get_ring_initialization(params, ring_counts)
        # Quick score check
        score = _compute_score(P)
        return -score
        
    try:
        # Use differential evolution for global search
        res = differential_evolution(neg_score, bounds, maxiter=maxiter, popsize=15, 
                                     tol=1e-6, seed=seed, polish=True, 
                                     workers=1,
                                     init='latinhypercube')
        
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
        
    except Exception:
        # Fallback to heuristic if DE fails or takes too long
        best_P = _get_hex_initial_points()
        
    return best_P


def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98


def _local_search(pts, n_iters=2000, seed=12345):
    """
    Local search: perturb one point at a time, accept if score improves.
    """
    rng = np.random.default_rng(seed)
    best_pts = pts.copy()
    best_score = _compute_score(best_pts)
    
    step_size = 0.01
    temperature = 0.003
    
    for it in range(n_iters):
        idx = rng.integers(0, N)
        old_pos = best_pts[idx].copy()
        new_pos = old_pos + step_size * rng.normal(size=2)
        
        # Keep inside container
        new_r = np.sqrt(np.sum(new_pos ** 2))
        if new_r > 0.99:
            new_pos *= 0.99 / new_r
        
        test_pts = best_pts.copy()
        test_pts[idx] = new_pos
        new_score = _compute_score(test_pts)
        
        delta = new_score - best_score
        if delta > 0 or rng.random() < np.exp(delta / temperature):
            best_pts[idx] = new_pos
            if new_score > best_score:
                best_score = new_score
        
        temperature *= 0.99995
        step_size = max(0.001, step_size * 0.99995)
    
    return best_pts, best_score


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # 5s buffer before 120s limit
    
    # Define ring structures to try
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
    ]
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    # Try each ring structure
    for ring_counts in ring_structures:
        if time.time() > deadline - 50.0:
            break
        
        # 1. Global search for ring parameters
        P_candidate = _global_search_ring_params(ring_counts, seed=42, maxiter=60)
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        # 2. Local refinement with binary search on r
        # Start with a slightly smaller r than the current score to ensure feasibility
        r_low = score_best * 0.95
        r_high = 0.15
        
        # Ensure r_low is feasible by refining
        P_opt, E = _optimize_positions(P_candidate, r_low, maxiter=200)
        if E < 1e-8:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt
                score_best = score
            r_low = score  # Update lower bound to the best found so far
        
        # Binary search on radius
        for _ in range(6):
            if time.time() > deadline - 15.0:
                break
                
            r_mid = (r_low + r_high) / 2.0
            
            # Initialize with the best P found so far
            P_init = P_best.copy()
            
            P_opt, E = _optimize_positions(P_init, r_mid, maxiter=150)
            
            if E < 1e-8:
                score = _compute_score(P_opt)
                if score > score_best:
                    P_best = P_opt
                    score_best = score
                r_low = r_mid
            else:
                r_high = r_mid
    
    # Final local search refinement
    if time.time() < deadline - 10.0:
        P_best, score_best = _local_search(P_best, n_iters=1500, seed=99999)
    
    # Final L-BFGS-B polish
    if time.time() < deadline - 5.0:
        r_target = score_best * 1.0001
        P_opt, E = _optimize_positions(P_best, r_target, maxiter=100)
        polished_score = _compute_score(P_opt)
        if polished_score > score_best:
            P_best = P_opt
            score_best = polished_score
    
    # Validate
    radii = np.sqrt(np.sum(P_best ** 2, axis=1))
    if np.any(~np.isfinite(P_best)) or np.any(radii >= 1.0):
        # Fallback
        P_best = _get_hex_initial_points()
    
    return P_best
