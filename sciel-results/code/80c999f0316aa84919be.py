"""
Packing 300 circles in a unit circle.
Strategy:
1. Initialize with hexagonal lattice.
2. Force-directed relaxation to find a valid packing for a target radius.
3. Binary search on radius.
4. High-precision polishing using soft-min gradient descent with adaptive beta.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    # Estimate spacing for hexagonal packing
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate hex lattice points
    max_k = int(1.0 / (s * 0.5)) + 3
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            # Standard hex lattice: offset every other row
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 0.999:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) >= n:
        # Select n closest to center
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices].copy()
    else:
        # Fallback: random points inside unit circle
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.95
    
    # Add small perturbation to break symmetry
    points += np.random.normal(0, 1e-5, points.shape)
    
    return points

def _force_directed_relax(points, r, time_limit=10.0, max_iter=5000):
    """
    Relax points to eliminate overlaps for a given radius r.
    Returns (relaxed_points, is_valid).
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    # Adaptive learning rate
    eta = 0.1
    min_eta = 1e-6
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        # Only consider pairs closer than 2r
        mask = (dists < 2.0 * r) & (dists > 1e-10)
        # Exclude diagonal
        np.fill_diagonal(mask, False)
        
        # Vectorized force calculation for overlapping pairs
        # Force magnitude proportional to overlap
        overlaps = 2.0 * r - dists
        overlaps[~mask] = 0.0
        
        # Unit vectors
        # dists is symmetric, diff is antisymmetric
        # We want to push i away from j if i is at i and j is at j
        # Force on i from j: direction (p_i - p_j) / dist
        # We can compute this using diff and dists
        # diff[i,j] = p_i - p_j
        # dists[i,j] = |p_i - p_j|
        
        # To avoid division by zero, use safe dists
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        
        # Force contribution from j on i: (overlap / dist) * (p_i - p_j)
        # This is (overlap / dist^2) * diff
        # We need to sum over j for each i
        
        # Create a matrix of coefficients
        coeff = overlaps / (safe_dists ** 2)
        coeff[~mask] = 0.0
        
        # Force on i is sum_j coeff[i,j] * diff[i,j]
        forces += np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary repulsion
        norms = np.linalg.norm(points, axis=1)
        boundary_violations = norms - (1.0 - r)
        boundary_mask = boundary_violations > 0
        
        if np.any(boundary_mask):
            # Force pushing inward
            # Direction is -p_i / |p_i|
            # Magnitude proportional to violation
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            boundary_forces = -boundary_violations[:, np.newaxis] * points / safe_norms[:, np.newaxis]
            forces[boundary_mask] += boundary_forces[boundary_mask]
        
        # Update positions
        points += eta * forces
        
        # Decay learning rate
        eta *= 0.999
        
        # Check validity
        min_pair_dist = np.min(dists[mask]) if np.any(mask) else np.inf
        max_norm = np.max(norms)
        
        if min_pair_dist >= 2.0 * r and max_norm <= 1.0 - r:
            return points, True
            
    # Final check
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    max_norm = np.max(np.linalg.norm(points, axis=1))
    
    is_valid = (min_pair_dist >= r) and (max_norm <= 1.0 - r)
    return points, is_valid

def _soft_min_and_grad(values, beta):
    """
    Compute soft-min and its gradient weights using log-sum-exp trick.
    """
    values = np.array(values)
    z = -beta * values
    
    # Log-sum-exp for numerical stability
    max_z = np.max(z)
    exp_vals = np.exp(z - max_z)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_z + np.log(sum_exp)
    
    soft_min_val = -log_sum_exp / beta
    weights = exp_vals / sum_exp
    
    return soft_min_val, weights

def _objective_and_grad(x, beta):
    """
    Compute the negative soft-min objective and its gradient.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Constraint values: d_ij/2 for pairs, 1-|p_i| for boundary
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft-min
    r_soft, weights = _soft_min_and_grad(all_vals, beta)
    objective = -r_soft
    
    # Gradient
    grad = np.zeros_like(p)
    
    # Pair contributions
    w_pairs = weights[:len(pair_vals)]
    i_idx, j_idx = iu
    safe_pair_dists = np.where(pair_vals > 1e-12, pair_vals, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    vecs = p[i_idx] - p[j_idx]
    
    np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    w_bounds = weights[len(pair_vals):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    # Gradient of objective = -gradient of soft_min
    grad_obj = -grad
    
    return objective, grad_obj

def _local_optimize(points, time_limit=30.0, beta_schedule=None, max_iter=500):
    """
    Local optimization using L-BFGS-B with soft-min objective and adaptive beta.
    """
    start_time = time.time()
    
    if beta_schedule is None:
        beta_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        try:
            # Use tighter tolerances for higher beta values
            if beta >= 10000:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-10,
                    'ftol': 1e-14,
                    'maxls': 100,
                    'disp': False
                }
            elif beta >= 1000:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-9,
                    'ftol': 1e-13,
                    'maxls': 80,
                    'disp': False
                }
            else:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 50,
                    'disp': False
                }
            
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options=opts
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
                
        except Exception:
            continue
    
    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization
    points = _hex_lattice_init(N)
    
    # 2. Estimate initial radius
    current_r = _compute_score(points)
    
    # Theoretical upper bound: 1/sqrt(N) is a rough lower bound for packing density ~0.9
    # For N=300, r ~ 0.053
    # Let's set a search range
    r_low = current_r
    r_high = 0.055  # Slightly above best known 0.05315
    
    # 3. Binary search for max valid r using Force-Directed Relaxation
    # We will do a few steps of binary search
    best_points = points.copy()
    best_r = current_r
    
    for _ in range(10):  # 10 binary search steps
        if time.time() > deadline - 40:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to relax for r_mid
        relaxed_points, is_valid = _force_directed_relax(points, r_mid, time_limit=2.0, max_iter=2000)
        
        if is_valid:
            r_low = r_mid
            best_points = relaxed_points
            best_r = r_mid
        else:
            r_high = r_mid
            
        # Update points to the relaxed version even if invalid, to provide a better start for next iteration?
        # Actually, if invalid, the relaxation might have improved the packing for a smaller r.
        # But we want to keep the best *valid* configuration.
        # Let's keep the last valid relaxed points as the starting point for the next search.
        # If invalid, we can still use the relaxed points as a starting point for the next (smaller) r.
        points = relaxed_points
    
    # 4. Polish the best found configuration using Soft-Min Optimization
    if time.time() < deadline - 20:
        remaining_time = min(20.0, deadline - 20 - time.time())
        # Use a beta schedule that starts moderate and goes high
        best_points, best_r = _local_optimize(
            best_points, 
            time_limit=remaining_time,
            beta_schedule=[100.0, 1000.0, 5000.0, 20000.0, 100000.0, 500000.0],
            max_iter=200
        )
        
    # 5. Final validation and clamping
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
        for i in range(len(best_points)):
            if norms[i] > 1.0:
                best_points[i] = best_points[i] / norms[i] * (1.0 - 1e-12)
        
    # Ensure all values are finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.9, neginf=-0.9)
        
    return best_points

# EVOLVE-BLOCK-END
