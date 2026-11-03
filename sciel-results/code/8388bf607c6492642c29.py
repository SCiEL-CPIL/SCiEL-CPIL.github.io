"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

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

def _get_ring_initialization(r_target, ring_radii=None, ring_angles=None):
    """
    Generate initial points based on concentric rings matching the known optimal structure.
    Ring structure: 1 + 6 + 12 + 18 + 13 = 50
    """
    P = np.zeros((N, 2))
    
    # Ring 0: 1 point at center
    P[0] = [0.0, 0.0]
    count = 1
    
    ring_counts = [6, 12, 18, 13]
    
    if ring_radii is None:
        # For hexagonal packing, ring k radius is approximately k * 2r * sqrt(3)/2
        # But let's use a better estimate: R_k ≈ k * 2r * 0.866 for hex packing
        # Actually, for the known optimal, let's use empirical ratios
        # R1/R0 ~ 2, R2/R0 ~ 3.4, R3/R0 ~ 4.8, R4/R0 ~ 5.5 (approximate)
        ring_radii = [2.0 * r_target, 3.4 * r_target, 4.8 * r_target, 5.5 * r_target]
    
    if ring_angles is None:
        ring_angles = [0.0, 0.0, 0.0, 0.0]  # Base angles
    
    for k, n_pts in enumerate(ring_counts):
        R = ring_radii[k]
        base_angle = ring_angles[k]
        angles = base_angle + np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        for i in range(n_pts):
            if count >= N:
                break
            P[count, 0] = R * np.cos(angles[i])
            P[count, 1] = R * np.sin(angles[i])
            count += 1
            
    return P

def _optimize_positions(P_init, r_target, max_iter=500, tol=1e-10, deadline=None):
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
        
    res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50})
    P_opt = res.x.reshape(N, 2)
    min_energy = objective(P_opt)
    return P_opt, min_energy

def _optimize_ring_params(r_target, P_init, max_iter=200, tol=1e-10):
    """
    Optimize ring radii and angles as parameters, then compute positions.
    This gives a more structured optimization.
    """
    # Parameters: 4 ring radii + 4 ring angles = 8 parameters
    # Initial values
    ring_counts = [6, 12, 18, 13]
    init_radii = [2.0 * r_target, 3.4 * r_target, 4.8 * r_target, 5.5 * r_target]
    init_angles = [0.0, 0.0, 0.0, 0.0]
    x0 = np.concatenate([init_radii, init_angles])
    
    def objective(params):
        radii = params[:4]
        angles = params[4:]
        P = _get_ring_initialization(r_target, radii, angles)
        return _energy(P, r_target)
    
    def grad(params):
        # Use numerical gradient for the ring parameters
        eps = 1e-7
        g = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            g[i] = (objective(params_plus) - f0) / eps
        return g
    
    res = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    
    best_params = res.x
    radii = best_params[:4]
    angles = best_params[4:]
    P = _get_ring_initialization(r_target, radii, angles)
    energy = _energy(P, r_target)
    
    return P, energy, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer
    
    # Start with the ring structure at a reasonable r
    # Known optimal is ~0.1258254895304
    r_start = 0.12
    
    # Get initial ring-based packing
    P_init = _get_ring_initialization(r_start)
    score_best = _compute_score(P_init)
    P_best = P_init.copy()
    
    # First, optimize positions for r_start
    if time.time() < deadline - 30.0:
        P_opt, E = _optimize_positions(P_init, r_start, max_iter=300, tol=1e-10)
        score = _compute_score(P_opt)
        if score > score_best:
            P_best = P_opt
            score_best = score
    
    # Binary search for optimal r
    r_low = r_start
    r_high = 0.14
    
    # Ensure r_low is feasible
    P_opt, E = _optimize_positions(P_best, r_low, max_iter=200, tol=1e-10)
    if E < 1e-8:
        score = _compute_score(P_opt)
        if score > score_best:
            P_best = P_opt
            score_best = score
    
    # Binary search iterations
    for iteration in range(15):
        if time.time() > deadline - 15.0:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Initialize with ring structure for r_mid
        P_ring = _get_ring_initialization(r_mid)
        
        # Optimize positions
        P_opt, E = _optimize_positions(P_ring, r_mid, max_iter=400, tol=1e-10)
        
        if E < 1e-8:
            # Feasible, try higher r
            r_low = r_mid
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt
                score_best = score
        else:
            # Infeasible, try lower r
            r_high = r_mid
    
    # Final polish: optimize at the best feasible r with tighter tolerance
    if time.time() < deadline - 5.0:
        # Use the ring structure at r_low as starting point
        P_ring = _get_ring_initialization(r_low)
        P_opt, E = _optimize_positions(P_ring, r_low, max_iter=1000, tol=1e-12)
        score = _compute_score(P_opt)
        if score > score_best:
            P_best = P_opt
            score_best = score
        
        # Also try optimizing from the current best
        P_opt2, E2 = _optimize_positions(P_best, r_low, max_iter=500, tol=1e-12)
        score2 = _compute_score(P_opt2)
        if score2 > score_best:
            P_best = P_opt2
            score_best = score2
    
    return P_best
# EVOLVE-BLOCK-END
