"""Improved program using Force-Directed Optimization with Simulated Annealing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_score(P):
    """
    Calculate the maximum radius r for a given set of points P.
    r = min( min_{i<j} dist(i,j)/2, min_i (1 - |P_i|) )
    """
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    # Distance to boundary
    dists = np.linalg.norm(P, axis=1)
    r_bound = np.min(1.0 - dists)
    
    # Pairwise distances
    # Using broadcasting for speed, though O(N^2)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _get_hex_initialization():
    """
    Generate initial points on a hexagonal lattice clipped to the unit circle.
    This provides a good structural starting point, much better than random or single ring.
    """
    points = []
    # Estimate spacing to get roughly N points
    # Area of unit circle is pi. Area of hex cell is ~ (sqrt(3)/2) * s^2.
    # We want density ~ N/pi.
    s = np.sqrt(np.pi / (N * np.sqrt(3) / 2))
    
    # Generate a grid of hex points
    # To ensure we cover the circle, we generate points in a square bounding box
    # and filter them.
    max_extent = 2.0
    steps = int(max_extent / s) + 2
    
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    P = np.array(points)
    # Keep only points inside the unit circle (with a small margin to be safe)
    dists = np.linalg.norm(P, axis=1)
    mask = dists < 0.95
    P = P[mask]
    
    if len(P) < N:
        # Fallback to random if hex lattice is too sparse (shouldn't happen with this s)
        return np.random.randn(N, 2) * 0.5
    else:
        # Select N points closest to the center to ensure a compact core
        # Actually, for packing, we want to fill the space. 
        # A better heuristic: Sort by distance to center and take the first N? 
        # No, that creates a cluster. 
        # Let's just take the first N from the list, but maybe randomize slightly to break symmetry.
        # Actually, the hex lattice is symmetric. Breaking symmetry is important.
        # Let's take a random subset of size N from the valid points? 
        # No, that might leave large holes.
        # Let's take the points that are "most central" but ensure we cover the boundary?
        # Simplest: Take the first N points generated. The generation order is row-major.
        # This will be a rectangular block of the hex lattice.
        # To make it circular, we should probably select points based on a radial sweep or just take all and if >N, keep the ones closest to center?
        # If we keep closest to center, we get a disk of points. This is a good start.
        idx = np.argsort(dists)
        P = P[idx[:N]]
        
    # Add small random jitter to break perfect symmetry
    P += np.random.randn(N, 2) * 0.01
    
    return P

def _force_directed_step(P, r, alpha, gamma):
    """
    Perform one step of force-directed relaxation.
    Returns new P and the maximum overlap (violation).
    """
    forces = np.zeros_like(P)
    
    # 1. Pairwise Repulsion
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # We want to push apart if dist < 2r
    # Force magnitude proportional to overlap
    overlap = np.maximum(0, 2*r - dist)
    
    # Unit vectors
    # Avoid division by zero
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 1e-10)
    
    # Force on i from j: direction (P_i - P_j) / dist * overlap
    # Sum over j
    # F_i = sum_j (P_i - P_j) / dist_ij * overlap_ij
    # Note: (P_i - P_j) is the diff matrix row i.
    # term_ij = diff[i,j] * overlap[i,j] / dist[i,j]
    # We can compute this as:
    # weight = overlap / dist
    # F_i = sum_j weight[i,j] * (P_i - P_j)
    # F_i = P_i * sum_j weight[i,j] - sum_j weight[i,j] * P_j
    
    weight = np.zeros_like(dist)
    np.divide(overlap, dist, out=weight, where=dist > 1e-10)
    
    sum_w = np.sum(weight, axis=1)
    term1 = P * sum_w[:, np.newaxis]
    term2 = (P.T @ weight).T
    
    forces += (term1 - term2) * gamma
    
    # 2. Boundary Constraint
    dist_origin = np.linalg.norm(P, axis=1)
    limit = 1.0 - r
    # If dist_origin > limit, push inward
    # Force is radial inward
    outside = dist_origin > limit
    if np.any(outside):
        viol = dist_origin[outside] - limit
        # Direction is -P / |P|
        unit_vec = P[outside] / dist_origin[outside, np.newaxis]
        forces[outside] -= unit_vec * viol * gamma * 1.0 # Boundary force is strong
    
    # 3. Centering Force (weak) to prevent drift
    forces -= 0.001 * P * gamma
    
    # Update
    P_new = P + alpha * forces
    
    # Calculate max violation for annealing logic
    # Re-compute distances for the new P to check if it's valid
    # Actually, we can estimate violation from the forces or just re-compute score
    # For speed, we'll rely on the score calculation in the main loop.
    
    return P_new

def _optimize_with_force_directed(P_init, time_limit):
    """
    Main optimization loop using force-directed dynamics and annealing.
    """
    P = P_init.copy()
    r_curr = 0.05 # Start small
    r_best = _compute_score(P)
    P_best = P.copy()
    
    start_time = time.time()
    
    # Parameters
    alpha = 0.05      # Learning rate / step size
    gamma = 10.0      # Force strength multiplier
    r_step = 0.001    # Increase in r per successful relaxation
    r_backoff = 0.0005 # Decrease in r if stuck
    max_stall = 20    # Number of iterations to try at same r before backoff/stall
    stall_count = 0
    
    # Initial relaxation at starting r
    for _ in range(50):
        P = _force_directed_step(P, r_curr, alpha, gamma)
    
    r_curr = _compute_score(P)
    if r_curr > r_best:
        r_best = r_curr
        P_best = P.copy()
    
    while time.time() - start_time < time_limit - 10: # Leave 10s for polish
        # Try to increase r
        r_target = r_curr + r_step
        
        # Relax at r_target
        success = False
        for _ in range(100): # Max iterations to relax at this r
            P = _force_directed_step(P, r_target, alpha, gamma)
            
            # Check if stable
            # We can check if the change in P is small, or just compute score
            score_now = _compute_score(P)
            
            # If score_now is close to r_target, it's stable enough
            if score_now >= r_target * 0.999:
                success = True
                break
            # If score dropped significantly, maybe we are in a bad state
            # But force-directed usually keeps score non-decreasing if alpha is small?
            # Actually, it can oscillate.
            
            # Time check inside loop
            if time.time() - start_time > time_limit - 10:
                return P_best, r_best
        
        if success:
            r_curr = r_target
            stall_count = 0
            if r_curr > r_best:
                r_best = r_curr
                P_best = P.copy()
            # Reduce step size as we get closer to optimum?
            # r_step = max(0.0001, r_step * 0.99)
        else:
            # Failed to relax at r_target
            stall_count += 1
            if stall_count > max_stall:
                # Back off r
                r_curr = max(0.05, r_curr - r_backoff)
                stall_count = 0
                # Add some random kick to escape local minimum
                P += np.random.randn(N, 2) * 0.01 * r_curr
                # Re-relax
                for _ in range(50):
                    P = _force_directed_step(P, r_curr, alpha, gamma)
                r_curr = _compute_score(P)
            else:
                # Try again with smaller alpha or just continue
                alpha *= 0.95
                if alpha < 0.001:
                    alpha = 0.05
                    r_curr = max(0.05, r_curr - r_backoff)
                    P += np.random.randn(N, 2) * 0.01 * r_curr
                    for _ in range(50):
                        P = _force_directed_step(P, r_curr, alpha, gamma)
                    r_curr = _compute_score(P)
    
    return P_best, r_best

def _local_polish(P, time_limit):
    """
    Final high-precision polish using Nelder-Mead on the exact score.
    """
    P_curr = P.copy()
    score_curr = _compute_score(P_curr)
    
    def neg_score(x):
        return -_compute_score(x.reshape(N, 2))
    
    try:
        # Use Nelder-Mead with tight tolerances
        res = minimize(neg_score, P_curr.flatten(), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        P_opt = res.x.reshape(N, 2)
        score_opt = _compute_score(P_opt)
        
        if score_opt > score_curr:
            return P_opt, score_opt
    except Exception:
        pass
        
    return P_curr, score_curr

def construct():
    start_time = time.time()
    total_time_limit = 140.0 # Safety margin for 150s limit
    
    # 1. Initialization
    P_init = _get_hex_initialization()
    
    # 2. Force-Directed Optimization
    # Allocate most time to this phase
    fd_time_limit = total_time_limit - 20.0
    P_fd, r_fd = _optimize_with_force_directed(P_init, fd_time_limit)
    
    # 3. Local Polish
    if time.time() - start_time < total_time_limit - 5.0:
        P_best, r_best = _local_polish(P_fd, total_time_limit - 5.0)
    else:
        P_best, r_best = P_fd, r_fd
        
    return P_best

# EVOLVE-BLOCK-END
