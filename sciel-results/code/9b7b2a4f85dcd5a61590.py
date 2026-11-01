"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for the given centers."""
    if len(centers) < 2:
        return np.inf
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_bound_dist = np.min(np.minimum(dist_x, dist_y))
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def _violation_objective(x, r):
    """Objective: sum of squared violations for radius r.
    Violation = max(0, r - d_ij/2)^2 for pairs, max(0, r - b_i)^2 for boundaries.
    """
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For each pair, violation is max(0, r - d/2)^2
    pair_violations = np.maximum(0, r - dists / 2.0) ** 2
    # Sum over unique pairs (upper triangle)
    iu = np.triu_indices(n, k=1)
    pair_viol_sum = np.sum(pair_violations[iu])
    
    # Boundary violations
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_dists = np.minimum(dist_x, dist_y)
    bound_violations = np.maximum(0, r - bound_dists) ** 2
    bound_viol_sum = np.sum(bound_violations)
    
    return pair_viol_sum + bound_viol_sum

def _violation_grad(x, r):
    """Gradient of violation objective."""
    centers = x.reshape(-1, 2)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For pair (i,j) with d < 2r: violation = (r - d/2)^2
    # dV/dc_i = 2*(r - d/2) * (-1/2) * (c_i - c_j)/d = -(r - d/2) * (c_i - c_j)/d
    # dV/dc_j = (r - d/2) * (c_i - c_j)/d
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < 2 * r and d > 1e-12:
                v = r - d / 2.0
                vec = diff[i, j]
                grad[i] -= v * vec / d
                grad[j] += v * vec / d
    
    # Boundary gradient
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_dists = np.minimum(dist_x, dist_y)
    
    for i in range(n):
        if bound_dists[i] < r:
            v = r - bound_dists[i]
            if dist_x[i] <= dist_y[i]:
                # x is limiting
                if centers[i, 0] > 0:
                    grad[i, 0] += v
                elif centers[i, 0] < 0:
                    grad[i, 0] -= v
                else:
                    grad[i, 0] += v * np.random.choice([-1, 1])
            else:
                # y is limiting
                if centers[i, 1] > 0:
                    grad[i, 1] += v
                elif centers[i, 1] < 0:
                    grad[i, 1] -= v
                else:
                    grad[i, 1] += v * np.random.choice([-1, 1])
    
    return grad.flatten()

def _construct_initial_grid():
    """Construct a good initial grid packing."""
    # 7x7 grid gives 49 points with spacing 1/7
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    # Add 50th point: place it at a position that maximizes minimum distance
    # Try several candidate positions and pick the best
    candidates = []
    # Center of each cell in a finer grid
    fine_ticks = (np.arange(15) + 0.5) / 15 - 0.5
    for y in fine_ticks:
        for x in fine_ticks:
            candidates.append([x, y])
    candidates = np.array(candidates)
    
    best_pos = None
    best_min_dist = -np.inf
    for pos in candidates:
        dists = np.sqrt(np.sum((grid_49 - pos) ** 2, axis=1))
        min_d = np.min(dists)
        if min_d > best_min_dist:
            best_min_dist = min_d
            best_pos = pos
    
    centers = np.vstack([grid_49, best_pos])
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin from 150s limit
    
    def _time_remaining():
        return deadline - time.time()
    
    # Initial guess
    centers = _construct_initial_grid()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Binary search on radius with violation minimization
    r_low = best_r
    r_high = 0.08  # upper bound, best known is ~0.0714
    
    # First, try to improve from the initial grid
    # Do a few rounds of BFGS at the current best radius to get a good starting point
    for _ in range(3):
        if _time_remaining() < 15:
            break
        x0 = centers.flatten()
        res = minimize(
            _violation_objective,
            x0,
            args=(best_r,),
            jac=lambda x: _violation_grad(x, best_r),
            method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        new_centers = res.x.reshape(-1, 2)
        new_r = _compute_radius(new_centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers.copy()
            centers = new_centers
        else:
            break
    
    # Binary search
    max_iters = 25
    for it in range(max_iters):
        if _time_remaining() < 20:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible packing for radius r_mid
        x0 = best_centers.flatten()
        
        res = minimize(
            _violation_objective,
            x0,
            args=(r_mid,),
            jac=lambda x: _violation_grad(x, r_mid),
            method='BFGS',
            options={'maxiter': 300, 'gtol': 1e-5}
        )
        
        new_centers = res.x.reshape(-1, 2)
        new_violation = res.fun
        
        # Check if feasible (violation is very small)
        if new_violation < 1e-6:
            # Feasible: try larger radius
            r_low = r_mid
            actual_r = _compute_radius(new_centers)
            if actual_r > best_r:
                best_r = actual_r
                best_centers = new_centers.copy()
        else:
            # Not feasible: try smaller radius
            r_high = r_mid
        
        # Update best based on actual computed radius
        actual_r = _compute_radius(best_centers)
        if actual_r > best_r:
            best_r = actual_r
    
    # Final optimization at the best radius found
    if _time_remaining() > 10:
        r_final = best_r
        x0 = best_centers.flatten()
        res = minimize(
            _violation_objective,
            x0,
            args=(r_final,),
            jac=lambda x: _violation_grad(x, r_final),
            method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-7}
        )
        best_centers = res.x.reshape(-1, 2)
        actual_r = _compute_radius(best_centers)
        if actual_r > best_r:
            best_r = actual_r
    
    # Ensure validity: strictly inside the square
    eps = 1e-9
    best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
    
    return best_centers
# EVOLVE-BLOCK-END
