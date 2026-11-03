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

def _smooth_score(P, r_target, beta=100.0):
    """
    Smooth surrogate for the score using log-sum-exp.
    We want to maximize r such that all pairwise dists >= 2r and all boundary dists >= r.
    This is equivalent to minimizing the maximum violation.
    Violation_cc_ij = max(0, 2*r_target - dist_ij)
    Violation_cb_i  = max(0, r_target - (1 - |p_i|)) = max(0, r_target + |p_i| - 1)
    
    We can define a smooth objective that penalizes violations.
    Actually, a better smooth objective for "maximize r" is to treat r as a variable?
    No, the current approach fixes r and tries to satisfy constraints.
    
    Let's use a different smooth objective:
    Minimize the sum of squared violations for a fixed r_target.
    Or, better, use the "soft-min" of the limiting distances.
    
    Let's stick to the epigraph idea from the directive.
    Maximize r subject to:
    dist_ij >= 2r  for all i<j
    1 - |p_i| >= r for all i
    
    This is a non-smooth problem.
    A smooth approximation:
    Objective: -r + lambda * (sum of soft-violations)
    
    Alternatively, we can optimize the positions P for a fixed r, minimizing the energy:
    E(P, r) = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r + |p_i| - 1)^2
    This is smooth (C1) and differentiable almost everywhere. The gradient is well-defined except at the kinks.
    We can use scipy.optimize.minimize on this energy for a fixed r.
    If the minimum energy is 0, then r is feasible.
    We can binary search on r.
    
    However, binary search on r might be slow if we do many full optimizations.
    
    Let's combine the parametric ring structure (inspiration 2) with the local refinement (current).
    The ring structure provides a very good initial guess close to the optimal.
    Then we use a general local optimizer to polish it.
    
    The current program's `_local_refine` uses a force-directed method which is essentially gradient descent on the energy.
    We can improve this by using scipy's L-BFGS-B or Powell on the energy function.
    """
    pass

def _energy(P, r):
    """
    Smooth energy function: sum of squared overlaps.
    This is differentiable, allowing gradient-based optimization.
    """
    # Circle-circle overlaps
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Avoid division by zero in gradient calculation later if needed, but here we just compute value.
    # For the value, max(0, x)^2 is fine.
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    
    # Circle-boundary overlaps
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))

def _energy_grad(P, r):
    """
    Gradient of the energy function.
    E = 0.5 * sum (max(0, 2r - d_ij)^2 + max(0, r + |p_i| - 1)^2)
    
    dE/dP_i = sum_{j != i} dE/d(d_ij) * d(d_ij)/dP_i + dE/d(|p_i|) * d(|p_i|)/dP_i
    
    Let V_cc_ij = max(0, 2r - d_ij)
    dE/d(d_ij) = -V_cc_ij (since d(V^2)/dd = 2V * dV/dd, dV/dd = -1, so -2V, but we have 0.5 factor, so -V)
    Wait, E_cc = 0.5 * V_cc_ij^2. dE_cc/dd_ij = V_cc_ij * dV_cc_ij/dd_ij = V_cc_ij * (-1) = -V_cc_ij.
    d(d_ij)/dP_i = (P_i - P_j) / d_ij
    
    So contribution to grad_i from pair (i,j) is: -V_cc_ij * (P_i - P_j) / d_ij
    
    For boundary:
    V_cb_i = max(0, r + |p_i| - 1)
    E_cb_i = 0.5 * V_cb_i^2
    dE_cb_i/d(|p_i|) = V_cb_i
    d(|p_i|)/dP_i = P_i / |p_i|
    
    So contribution to grad_i from boundary is: V_cb_i * P_i / |p_i|
    """
    grad = np.zeros_like(P)
    
    # Pairwise
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    V_cc = np.maximum(2.0 * r - dist, 0.0)
    # We need to compute: sum_j -V_cc_ij * (P_i - P_j) / d_ij
    # This is equivalent to: sum_j V_cc_ij * (P_j - P_i) / d_ij
    # Let W_ij = V_cc_ij / d_ij
    # Grad_i = sum_j W_ij * (P_j - P_i)
    #         = sum_j W_ij P_j - P_i sum_j W_ij
    
    # Handle division by zero
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    W = V_cc * inv_dist
    # W is (N, N). Diagonal is 0 because V_cc is 0 on diagonal (dist=inf -> max(0, 2r-inf)=0)
    
    sum_W = np.sum(W, axis=1) # (N,)
    term1 = (P.T @ W).T # (N, 2) = sum_j W_ij P_j
    term2 = P * sum_W[:, np.newaxis] # (N, 2) = P_i sum_j W_ij
    
    grad += term1 - term2
    
    # Boundary
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r + dist_origin - 1.0, 0.0)
    
    # dE/dP_i = V_cb_i * P_i / |p_i|
    # Handle |p_i| = 0
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad

def _get_ring_initialization(r_target):
    """
    Generate initial points based on concentric rings.
    Ring 0: 1 point at center
    Ring 1: 6 points
    Ring 2: 12 points
    Ring 3: 18 points
    Ring 4: 13 points (remaining)
    Total: 50
    
    Radii are approximately k * r_target * sqrt(3) for hexagonal packing?
    Actually, for hex packing, distance between centers is 2r.
    Ring k has distance approx k * 2r * cos(pi/6)? No.
    Distance from center to ring k center is approx k * 2r * sqrt(3)/2 * 2/3?
    Let's just use k * 2 * r_target * 0.866 (approx sqrt(3)/2 * 2r? No, 2r is side length).
    Distance between adjacent centers is 2r.
    In a hex grid, the distance from the origin to the k-th ring is k * 2r * sqrt(3) / 2 * 2?
    Let's use the known structure:
    R1 ~ 2r
    R2 ~ 3r? No, 2 * 2r * cos(30) = 2r * sqrt(3) ~ 3.46r?
    
    Let's use a simple heuristic: R_k = k * 2 * r_target * 0.9
    Or better, optimize the ring radii.
    For now, a good guess is:
    R_1 = 2 * r_target
    R_2 = 3 * r_target * np.sqrt(3) / 2? 
    Let's just use R_k = k * 2 * r_target * 0.85
    """
    P = np.zeros((N, 2))
    
    # Ring 0
    P[0] = [0.0, 0.0]
    count = 1
    
    ring_counts = [6, 12, 18, 13]
    
    for k, n_pts in enumerate(ring_counts, 1):
        R = k * 2 * r_target * 0.85
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        # Add a small offset to break symmetry if needed, or just start at 0
        for i in range(n_pts):
            if count >= N:
                break
            P[count, 0] = R * np.cos(angles[i])
            P[count, 1] = R * np.sin(angles[i])
            count += 1
            
    return P

def _optimize_for_r(P_init, r_target, max_iter=500, tol=1e-8):
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
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
    P_opt = res.x.reshape(N, 2)
    min_energy = objective(P_opt)
    return P_opt, min_energy

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s buffer
    
    # Start with a reasonable guess for r
    # Known optimal is ~0.1258
    # Let's start with 0.12 and binary search up.
    
    # First, get a valid packing with a small r to ensure we have a starting point
    # Use the hex lattice initializer from the current program as a fallback
    def _get_hex_initial_points():
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

    P_init = _get_hex_initial_points()
    score_best = _compute_score(P_init)
    P_best = P_init.copy()
    
    # If the hex lattice is already good, use it.
    # Otherwise, try the ring structure.
    
    # Let's try to find a better r using binary search with the ring initializer.
    # We want to find the largest r such that min_energy(P_opt, r) < eps.
    
    r_low = 0.1
    r_high = 0.15
    
    # Ensure r_low is feasible
    # Try to optimize for r_low
    P_ring = _get_ring_initialization(r_low)
    P_opt, E = _optimize_for_r(P_ring, r_low, max_iter=200)
    if E < 1e-6:
        # Feasible
        score = _compute_score(P_opt)
        if score > score_best:
            P_best = P_opt
            score_best = score
            
    # Binary search
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Initialize with ring structure for r_mid
        P_ring = _get_ring_initialization(r_mid)
        
        # Optimize
        P_opt, E = _optimize_for_r(P_ring, r_mid, max_iter=300)
        
        if E < 1e-6:
            # Feasible, try higher r
            r_low = r_mid
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt
                score_best = score
        else:
            # Infeasible, try lower r
            r_high = r_mid
            
    # Final polish: use the best P found and refine it with a local optimizer on the score itself?
    # Or just do one more optimization at r_low (the best feasible r found)
    
    if time.time() < deadline - 5.0:
        P_ring = _get_ring_initialization(r_low)
        P_opt, E = _optimize_for_r(P_ring, r_low, max_iter=1000, tol=1e-10)
        score = _compute_score(P_opt)
        if score > score_best:
            P_best = P_opt
            score_best = score
            
    return P_best
# EVOLVE-BLOCK-END
