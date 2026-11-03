"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Minimum distance to boundary
    min_boundary = 0.5 - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    return min(min_boundary, min_pair)


def _energy_and_grad(centers, r_target):
    """
    Smooth surrogate energy: penalize pairwise overlaps and wall violations.
    Uses soft-min via log-sum-exp for smoothness.
    Returns (energy, gradient).
    """
    n = len(centers)
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=-1)
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)

    # Soft-min of (dists - 2*r_target) with temperature beta
    beta = 100.0
    # We want to penalize when dists < 2*r_target
    # Use soft-min: -1/beta * log(sum(exp(-beta*(dists - 2*r_target))))
    # But this is expensive for all pairs. Instead, use a simpler approach:
    # Sum of max(0, 2*r_target - dists)^2 for all pairs (smooth approximation)
    
    # Actually, let's use a different smooth energy:
    # E_pair = sum_{i<j} max(0, 2*r_target - d_ij)^2
    # This is not smooth at d_ij = 2*r_target, but we can smooth it.
    
    # Better: use E = sum_{i<j} (max(0, 2*r_target - d_ij))^2 
    # For gradient, we need to be careful.
    
    # Let's use a simpler but effective approach:
    # Energy = sum over all pairs of (2*r_target - d_ij)^2 if d_ij < 2*r_target
    # Plus wall penalties
    
    # For smoothness, use: E_pair = sum_{i<j} exp(-beta*(d_ij - 2*r_target)) for d_ij < 2*r_target
    # This is still not smooth. 
    
    # Let's use the standard approach: minimize sum of squared violations
    # with a smooth approximation.
    
    # Simple approach: E = sum_{i<j} max(0, 2*r_target - d_ij)^2 + wall_penalties
    # Gradient is piecewise.
    
    # For BFGS to work well, let's use a smooth soft-min:
    # E_soft = -1/beta * log(sum_{i<j} exp(-beta * (d_ij - 2*r_target)))
    # This approximates min_{i<j} (d_ij - 2*r_target) as beta -> inf
    # But we want to penalize ALL overlaps, not just the worst one.
    
    # Let's use: E = sum_{i<j} max(0, 2*r_target - d_ij)^2
    # and compute gradient numerically or analytically.
    
    # Analytical gradient for E_pair = sum_{i<j} (max(0, 2*r_target - d_ij))^2:
    # dE/dcenters_i = sum_{j != i} 2 * max(0, 2*r_target - d_ij) * (centers_i - centers_j) / d_ij
    
    # Wall penalty: E_wall = sum_i sum_k max(0, r_target - (0.5 - |c_ik|))^2
    # = sum_i sum_k max(0, r_target + |c_ik| - 0.5)^2
    
    # Compute pairwise part
    violations = np.maximum(0, 2 * r_target - dists)
    energy_pair = np.sum(violations ** 2)
    
    # Gradient of pairwise part
    grad = np.zeros_like(centers)
    mask = violations > 0
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i, j]:
                diff = centers[i] - centers[j]
                d = dists[i, j]
                if d > 1e-10:
                    factor = 2 * violations[i, j] * diff / d
                    grad[i] += factor
                    grad[j] -= factor
    
    # Wall part
    abs_c = np.abs(centers)
    wall_viol = np.maximum(0, r_target + abs_c - 0.5)
    energy_wall = np.sum(wall_viol ** 2)
    
    # Gradient of wall part: d/dc_ik max(0, r_target + |c_ik| - 0.5)^2
    # = 2 * max(0, r_target + |c_ik| - 0.5) * sign(c_ik)
    grad_wall = 2 * wall_viol * np.sign(centers)
    grad += grad_wall
    
    total_energy = energy_pair + energy_wall
    return total_energy, grad


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin

    # Start with a grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()

    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)

    # Use iterative radius increase with BFGS optimization
    np.random.seed(42)
    
    current_radius = best_radius * 0.95  # Start slightly below current best
    max_radius_attempts = 30
    r_increase_factor = 1.01
    
    for attempt in range(max_radius_attempts):
        if time.time() > deadline - 10:
            break
        
        target_r = current_radius * r_increase_factor
        if target_r > 0.12:
            break
        
        # Try to find a configuration with radius >= target_r
        # Minimize energy with r_target = target_r
        x0 = best_centers.flatten()
        
        def objective(x):
            c = x.reshape(N, 3)
            e, _ = _energy_and_grad(c, target_r)
            return e
        
        def gradient(x):
            c = x.reshape(N, 3)
            _, g = _energy_and_grad(c, target_r)
            return g.flatten()
        
        try:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                            bounds=[(-0.5, 0.5)] * (N * 3),
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            continue
        
        new_centers = result.x.reshape(N, 3)
        new_radius = _compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
            current_radius = new_radius
        else:
            # Failed to increase, try a smaller step
            r_increase_factor *= 0.9
            if r_increase_factor < 1.001:
                break
    
    # Final polish with smaller radius increases
    if time.time() < deadline - 5:
        polish_r = best_radius
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            target_r = polish_r * 1.005
            x0 = best_centers.flatten()
            
            def objective_p(x):
                c = x.reshape(N, 3)
                e, _ = _energy_and_grad(c, target_r)
                return e
            
            def gradient_p(x):
                c = x.reshape(N, 3)
                _, g = _energy_and_grad(c, target_r)
                return g.flatten()
            
            try:
                result = minimize(objective_p, x0, jac=gradient_p, method='L-BFGS-B',
                                bounds=[(-0.5, 0.5)] * (N * 3),
                                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            except Exception:
                break
            
            new_centers = result.x.reshape(N, 3)
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
                polish_r = new_radius
            else:
                break
    
    return best_centers
# EVOLVE-BLOCK-END
