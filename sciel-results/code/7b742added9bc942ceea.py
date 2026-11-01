"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_score(pts):
    """Compute the packing score for given points."""
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary_dist = np.min(1.0 - radii)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r


def _feasibility_obj_and_grad(pts, R):
    """
    Objective: Minimize the sum of squared violations of constraints:
    1. d_ij >= 2R
    2. |c_i| <= 1 - R
    """
    n = len(pts)
    obj = 0.0
    grad = np.zeros_like(pts)
    
    # Pairwise constraints: d_ij >= 2R
    # d_ij = ||p_i - p_j||
    # Violation v_ij = max(0, 2R - d_ij)
    # Contribution to obj: v_ij^2
    # Gradient of v_ij^2 w.r.t p_i:
    # If d_ij < 2R:
    #   d(v_ij^2)/dp_i = 2 * v_ij * d(2R - d_ij)/dp_i = 2 * (2R - d_ij) * (-(p_i - p_j)/d_ij)
    #   = 2 * (2R - d_ij) * (p_j - p_i) / d_ij
    # If d_ij >= 2R: 0
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            if d < 2 * R:
                violation = 2 * R - d
                # grad obj += 2 * violation * (p_j - p_i) / d
                grad_factor = 2.0 * violation / d
                grad[i] += grad_factor * (pts[j] - pts[i])
                grad[j] += grad_factor * (pts[i] - pts[j])
                obj += violation ** 2
    
    # Boundary constraints: |c_i| <= 1 - R
    # Violation v_i = max(0, |c_i| - (1 - R))
    # Contribution: v_i^2
    # Gradient:
    # If |c_i| > 1 - R:
    #   d(v_i^2)/dp_i = 2 * v_i * d(|c_i|)/dp_i = 2 * (|c_i| - (1-R)) * (p_i / |c_i|)
    
    for i in range(n):
        r_i = np.linalg.norm(pts[i])
        limit = 1.0 - R
        if r_i > 1e-10:
            if r_i > limit:
                violation = r_i - limit
                grad_factor = 2.0 * violation / r_i
                grad[i] += grad_factor * pts[i]
                obj += violation ** 2
        else:
            # If at origin, gradient is 0 for norm, but we should push it out?
            # Actually if r_i < limit, no violation.
            pass
            
    return obj, grad


def _optimize_for_radius(pts0, R, time_budget=5.0):
    """
    Try to find a configuration with radius R.
    Returns (success, best_pts, min_violation).
    """
    start_time = time.time()
    pts = pts0.copy()
    
    # Initial projection to ensure inside circle
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    limit = 1.0 - R
    mask = radii > limit
    if np.any(mask):
        pts[mask] = pts[mask] * (limit / radii[mask, np.newaxis])
        
    best_pts = pts.copy()
    best_obj, _ = _feasibility_obj_and_grad(pts, R)
    
    # Use L-BFGS-B
    try:
        result = minimize(
            lambda x: _feasibility_obj_and_grad(x, R)[0],
            pts,
            jac=lambda x: _feasibility_obj_and_grad(x, R)[1],
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        pts = result.x
        
        # Project back
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        mask = radii > limit
        if np.any(mask):
            pts[mask] = pts[mask] * (limit / radii[mask, np.newaxis])
            
        obj, _ = _feasibility_obj_and_grad(pts, R)
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
            
    except Exception:
        pass
        
    # Check if feasible (obj close to 0)
    # Note: obj is sum of squares, so 0 means all violations are 0
    threshold = 1e-6
    success = best_obj < threshold
    
    return success, best_pts, best_obj


def _binary_search_radius(pts_init, R_low, R_high, time_budget=60.0):
    """
    Binary search for the maximum feasible radius.
    """
    start_time = time.time()
    best_pts = pts_init.copy()
    best_R = R_low
    
    # We will do a few iterations of binary search
    # Each iteration calls _optimize_for_radius
    # We allocate time per iteration
    
    while time.time() - start_time < time_budget:
        if R_high - R_low < 1e-6:
            break
            
        R_mid = (R_low + R_high) / 2.0
        
        # Try to optimize for R_mid
        # Allocate half the remaining time to this attempt?
        # Simpler: just run with a small fixed budget per step
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2.0:
            break
            
        success, pts_candidate, _ = _optimize_for_radius(pts_init, R_mid, time_budget=min(5.0, remaining * 0.5))
        
        if success:
            best_R = R_mid
            best_pts = pts_candidate.copy()
            R_low = R_mid
        else:
            R_high = R_mid
            
    return best_R, best_pts


def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    n_points = N
    
    # Initial guess for radius
    # Area argument: 50 * pi * r^2 <= pi * 1^2 => r <= 1/sqrt(50) ~ 0.141
    # Known best is ~0.1258
    R_low = 0.120
    R_high = 0.145
    
    # Initial points
    # Use a hexagonal lattice or random as starting point for the optimizer
    # Random is often better for escaping local minima in penalty methods
    np.random.seed(42)
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    radii = np.sqrt(np.random.uniform(0, 1, n_points))
    pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Run binary search
    best_R, best_pts = _binary_search_radius(pts_init, R_low, R_high, time_budget=60.0)
    
    # If we have time left, try to polish the best found solution with a very tight tolerance
    # Or try a few more random restarts with the binary search
    if time.time() < deadline - 10.0:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 5.0:
                break
            np.random.seed(seed)
            angles = np.random.uniform(0, 2 * np.pi, n_points)
            radii = np.sqrt(np.random.uniform(0, 1, n_points))
            pts_init_2 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            # Quick binary search
            R_lo_2 = best_R
            R_hi_2 = best_R + 0.01
            R_2, pts_2 = _binary_search_radius(pts_init_2, R_lo_2, R_hi_2, time_budget=min(10.0, deadline - time.time() - 5.0))
            
            if R_2 > best_R:
                best_R = R_2
                best_pts = pts_2
    
    # Final check: ensure points are valid
    # Compute score to be sure
    score = _compute_score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
