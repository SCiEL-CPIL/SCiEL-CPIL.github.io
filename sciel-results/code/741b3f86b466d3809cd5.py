"""Packing 50 equal circles in a unit circle using smooth energy minimization with ring structures."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # safety margin before 150s kill


def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    P = np.asarray(P)
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return 0.0
    
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return max(0.0, min(r_pair, r_bound))


def _energy(P, r):
    """
    Smooth energy function: sum of squared overlaps.
    This is differentiable, allowing gradient-based optimization.
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
            R = max(0.0, radii[i])
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
            
    return P


def _optimize_positions(P_init, r_target, time_limit=5.0):
    """
    Optimize positions P to minimize energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())


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


def _optimize_ring_structure(radii0, counts, angles_offset0=None, time_limit=30.0):
    """Optimize ring radii and angles for fixed counts using energy minimization."""
    if angles_offset0 is None:
        angles_offset0 = [0.0] * len(counts)
    
    n_rings = len(counts)
    
    def build_and_score(params):
        radii = np.maximum(params[:n_rings], 0.0)
        angles = params[n_rings:]
        P = _get_ring_initialization(np.concatenate([radii, angles]), counts)
        return _compute_score(P)
    
    x0 = np.concatenate([radii0, angles_offset0])
    
    t_start = time.time()
    try:
        res = minimize(
            lambda x: -build_and_score(x),
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success:
            radii_opt = np.maximum(res.x[:n_rings], 0.0)
            angles_opt = res.x[n_rings:]
            return radii_opt, angles_opt
    except Exception:
        pass
    
    return radii0, angles_offset0


def _polish(P, r_target, time_limit=10.0):
    """
    Final high-precision polish pass with tighter tolerances.
    Uses the smooth energy formulation with very tight convergence criteria.
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()
        
    try:
        res = minimize(objective, P.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        P_opt = res.x.reshape(N, 2)
        return P_opt
    except Exception:
        return P


def construct():
    t_start = time.time()
    
    # Define ring structures for N=50
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 10, 12, 21],
    ]
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    # For each ring structure, optimize and refine
    for counts in ring_structures:
        if time.time() - t_start > DEADLINE - 20:
            break
        
        n_rings = len(counts)
        
        # Initial radii: evenly spaced
        if n_rings == 1:
            radii0 = np.array([0.0])
        else:
            radii0 = np.linspace(0, 0.85, n_rings)
            radii0[0] = 0.0
        
        angles0 = [0.0] * n_rings
        
        # Optimize ring structure parameters
        radii_opt, angles_opt = _optimize_ring_structure(
            radii0, counts, angles0,
            time_limit=min(15.0, DEADLINE - (time.time() - t_start) - 25)
        )
        
        # Build centers from optimized ring structure
        P_candidate = _get_ring_initialization(
            np.concatenate([radii_opt, angles_opt]), counts
        )
        
        score_candidate = _compute_score(P_candidate)
        
        # Binary search on r with energy minimization
        if score_candidate > score_best:
            P_best = P_candidate.copy()
            score_best = score_candidate
        
        # Try to improve with binary search on r
        r_low = score_best * 0.95
        r_high = 0.14
        
        # Ensure r_low is feasible
        P_opt, E = _optimize_positions(P_candidate, r_low, time_limit=5.0)
        if E < 1e-6:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt.copy()
                score_best = score
            r_low = score
        
        # Binary search for maximum feasible r
        for _ in range(12):
            if time.time() - t_start > DEADLINE - 20:
                break
            r_mid = 0.5 * (r_low + r_high)
            P_opt, E = _optimize_positions(P_candidate, r_mid, time_limit=5.0)
            if E < 1e-6:
                score = _compute_score(P_opt)
                if score > score_best:
                    P_best = P_opt.copy()
                    score_best = score
                r_low = r_mid
            else:
                r_high = r_mid
    
    # Final polish pass with tighter tolerances
    if time.time() - t_start < DEADLINE - 10:
        r_target = score_best * 1.001
        P_polished = _polish(P_best, r_target, time_limit=15.0)
        score_polished = _compute_score(P_polished)
        if score_polished > score_best:
            P_best = P_polished
            score_best = score_polished
    
    return P_best
