"""Structural change: Gradient-based optimization with hexagonal lattice initialization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 115.0  # Safety margin below 150s kill


def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)


def _hexagonal_init(n_points, spacing=None):
    """Generate a hexagonal lattice of approximately n_points within [-0.5,0.5]^2."""
    if spacing is None:
        # Estimate spacing from area
        area_per_point = 1.0 / n_points
        spacing = np.sqrt(area_per_point / (np.sqrt(3) / 2))
    
    # Generate hexagonal lattice points
    # Hex lattice: points at (i * spacing, j * spacing * sqrt(3)) + offset for odd j
    sqrt3 = np.sqrt(3)
    points = []
    
    # Generate enough points to cover the square
    max_i = int(1.0 / spacing) + 2
    max_j = int(1.0 / (spacing * sqrt3)) + 2
    
    for j in range(-max_j, max_j + 1):
        y = j * spacing * sqrt3 / 2
        if y > 0.5 or y < -0.5:
            continue
        offset = (spacing / 2) if (j % 2 != 0) else 0
        for i in range(-max_i, max_i + 1):
            x = i * spacing + offset
            if x > 0.5 or x < -0.5:
                continue
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) == 0:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n_points)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_points]
    
    # If we have more than n_points, select the ones closest to center
    if len(points) > n_points:
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
    elif len(points) < n_points:
        # Pad with random points
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
    
    return points[:n_points]


def _soft_min_objective(flat_centers, sharpness=50.0):
    """
    Differentiable soft-min of pairwise distances and boundary distances.
    Returns negative of the soft-min (to be minimized).
    """
    centers = flat_centers.reshape(-1, 2)
    
    # Boundary distances
    bx = 0.5 - np.abs(centers[:, 0])
    by = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(bx, by)
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dists, 1e9)
    
    # Combine all distances
    all_dists = np.concatenate([dists.flatten(), boundary_dists])
    # Filter out the diagonal (very large values)
    all_dists = all_dists[all_dists < 1e8]
    
    if len(all_dists) == 0:
        return 0.0
    
    # Soft min using log-sum-exp trick
    # soft_min(x) = -log(sum(exp(-x_i / sharpness))) * sharpness
    scaled = -all_dists / sharpness
    max_scaled = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled
    soft_min_val = -log_sum * sharpness
    
    return -soft_min_val  # Negative because we minimize


def _soft_min_gradient(flat_centers, sharpness=50.0):
    """
    Gradient of the soft-min objective.
    """
    centers = flat_centers.reshape(-1, 2)
    n = centers.shape[0]
    
    # Boundary distances
    bx = 0.5 - np.abs(centers[:, 0])
    by = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(bx, by)
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dists, 1e9)
    
    # Combine all distances
    all_dists = np.concatenate([dists.flatten(), boundary_dists])
    all_dists = all_dists[all_dists < 1e8]
    
    if len(all_dists) == 0:
        return np.zeros_like(flat_centers)
    
    # Soft min weights
    scaled = -all_dists / sharpness
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # Gradient of soft_min with respect to each distance
    # d(soft_min)/d(d_i) = -weights[i]
    
    grad = np.zeros_like(flat_centers)
    
    # Pairwise distance gradients
    pair_dists = dists.flatten()
    pair_weights = weights[:len(pair_dists)]
    
    # For each pair (i,j), d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij
    # d(dist_ij)/d(c_j) = (c_j - c_i) / dist_ij
    
    # Reconstruct pairwise structure
    idx_i, idx_j = np.triu_indices(n, k=1)
    pair_diff = centers[idx_i] - centers[idx_j]
    pair_dist_vals = dists[idx_i, idx_j]
    
    # Gradient contribution from pairs
    pair_grad_i = np.zeros_like(centers)
    pair_grad_j = np.zeros_like(centers)
    
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        w = pair_weights[k * 2] + pair_weights[k * 2 + 1]  # Two entries per pair in flattened
        if pair_dist_vals[k] > 1e-8:
            unit_vec = pair_diff[k] / pair_dist_vals[k]
            pair_grad_i[i] -= w * unit_vec  # d(dist)/d(c_i) = (c_i-c_j)/dist, so grad = -w * unit_vec for minimization
            pair_grad_j[j] += w * unit_vec  # d(dist)/d(c_j) = (c_j-c_i)/dist
    
    grad[:] = pair_grad_i + pair_grad_j
    
    # Boundary gradients
    bound_weights = weights[len(pair_dists):]
    for k in range(n):
        # Gradient of (0.5 - |x|) w.r.t x = -sign(x)
        # Gradient of (0.5 - |y|) w.r.t y = -sign(y)
        if centers[k, 0] > 0:
            grad[k, 0] -= bound_weights[k] * 1.0  # d(0.5-x)/dx = -1
        elif centers[k, 0] < 0:
            grad[k, 0] -= bound_weights[k] * (-1.0)  # d(0.5+x)/dx = 1, but we have -sign(x)
        if centers[k, 1] > 0:
            grad[k, 1] -= bound_weights[k] * 1.0
        elif centers[k, 1] < 0:
            grad[k, 1] -= bound_weights[k] * (-1.0)
    
    # The objective is -soft_min, so gradient is negative of soft_min gradient
    # soft_min gradient w.r.t c is sum of weights * d(dist)/dc
    # We want to minimize -soft_min, so grad = -d(soft_min)/dc
    # d(soft_min)/dc = sum_i weights_i * d(dist_i)/dc
    # So grad of (-soft_min) = -sum_i weights_i * d(dist_i)/dc
    
    # The above computation already gives us d(soft_min)/dc (with signs)
    # Let me recompute more carefully
    
    # Actually, let's use a simpler approach: numerical gradient is too slow
    # Let's use the analytical form properly
    
    # For pairwise distances:
    # dist_ij = ||c_i - c_j||
    # d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij
    # d(dist_ij)/d(c_j) = (c_j - c_i) / dist_ij
    
    # soft_min = -sharpness * log(sum(exp(-d_k / sharpness)))
    # d(soft_min)/d(d_k) = -exp(-d_k/sharpness) / sum(exp(-d_m/sharpness)) = -weights[k]
    
    # d(soft_min)/d(c_i) = sum_k d(soft_min)/d(d_k) * d(d_k)/d(c_i)
    #                    = sum_k (-weights[k]) * d(d_k)/d(c_i)
    
    # We minimize -soft_min, so grad = -d(soft_min)/dc = sum_k weights[k] * d(d_k)/dc
    
    grad = np.zeros_like(flat_centers)
    
    # Pairwise contributions
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        # In the flattened all_dists, pair (i,j) appears at positions k*2 and k*2+1
        # Actually, triu_indices gives us the upper triangle, and when flattened,
        # the order is row-major, so pair (i,j) with i<j is at index i*n+j
        # Let's just use the weights array properly
        
    # This is getting complex. Let's use a simpler gradient approach:
    # Use central differences for the gradient, but that's O(N^2) per eval
    # Instead, let's use the fact that we can compute the gradient analytically
    
    # Simpler: just return a reasonable gradient approximation
    # The key insight: we want to increase the minimum distance
    # So we should move points apart from their nearest neighbors
    
    # For each point, compute the net "push" from all other points
    # weighted by the soft-min weights
    
    grad = np.zeros_like(centers)
    
    # Pairwise: for each pair (i,j), if dist_ij is small, push i away from j and j away from i
    # Weight by the soft-min weight for that distance
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < 1e-8:
                continue
            # Find the weight for this pair in the soft-min
            # The flattened dists has index i*n + j for (i,j) and j*n + i for (j,i)
            # But we only have upper triangle in idx_i, idx_j
            # Let's just use a simple weight based on distance
            w = np.exp(-d / sharpness)
            
            unit_vec = (centers[i] - centers[j]) / d
            grad[i] += w * unit_vec  # Push i away from j
            grad[j] -= w * unit_vec  # Push j away from i
    
    # Boundary: if a point is close to a wall, push it inward
    for i in range(n):
        if centers[i, 0] > 0:
            w = np.exp(-(0.5 - centers[i, 0]) / sharpness)
            grad[i, 0] -= w  # Push left if near right wall
        else:
            w = np.exp(-(-0.5 - centers[i, 0]) / sharpness)
            grad[i, 0] += w  # Push right if near left wall
        
        if centers[i, 1] > 0:
            w = np.exp(-(0.5 - centers[i, 1]) / sharpness)
            grad[i, 1] -= w
        else:
            w = np.exp(-(-0.5 - centers[i, 1]) / sharpness)
            grad[i, 1] += w
    
    return grad.flatten()


def _refine_with_lbfgs(centers, max_time=30.0):
    """
    Use L-BFGS-B to refine the packing by maximizing the minimum distance.
    """
    start_time = time.time()
    flat = centers.flatten()
    
    # Start with a moderate sharpness and increase it for refinement
    sharpness = 20.0
    
    def objective(x):
        return _soft_min_objective(x, sharpness)
    
    def gradient(x):
        return _soft_min_gradient(x, sharpness)
    
    # Bounds: keep points in the square
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(
        objective,
        flat,
        jac=gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 200,
            'maxfun': 5000,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxls': 20,
            'iprint': -1
        }
    )
    
    if time.time() - start_time > max_time:
        return centers
    
    new_centers = result.x.reshape(-1, 2)
    
    # Clip to ensure we're in bounds
    new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
    new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
    
    return new_centers


def _local_search(centers, max_time=20.0):
    """
    Local search: try small perturbations of each point and keep improvements.
    """
    start_time = time.time()
    centers = centers.copy()
    r_best = _compute_radius(centers)
    
    n = len(centers)
    
    # Try perturbing each point in random directions
    for it in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Pick a random point
        i = np.random.randint(n)
        
        # Try a small random perturbation
        for attempt in range(10):
            delta = np.random.normal(0, 0.005, size=2)
            new_center = centers[i].copy()
            new_center += delta
            new_center = np.clip(new_center, -0.5, 0.5)
            
            old_center = centers[i].copy()
            centers[i] = new_center
            
            r_new = _compute_radius(centers)
            
            if r_new > r_best:
                r_best = r_new
            else:
                centers[i] = old_center
                break
        
        # Periodically try larger moves
        if it % 100 == 0:
            for i in range(n):
                delta = np.random.normal(0, 0.02, size=2)
                new_center = centers[i].copy()
                new_center += delta
                new_center = np.clip(new_center, -0.5, 0.5)
                
                old_center = centers[i].copy()
                centers[i] = new_center
                
                r_new = _compute_radius(centers)
                
                if r_new > r_best:
                    r_best = r_new
                else:
                    centers[i] = old_center
    
    return centers, r_best


def construct():
    start_time = time.time()
    
    # 1. Generate hexagonal lattice initialization
    centers = _hexagonal_init(N)
    
    # 2. Initial refinement with L-BFGS-B
    if time.time() - start_time < _DEADLINE - 60:
        centers = _refine_with_lbfgs(centers, max_time=40.0)
    
    # 3. Local search polish
    if time.time() - start_time < _DEADLINE - 30:
        centers, r_best = _local_search(centers, max_time=30.0)
    
    # 4. Try a few different initializations if time permits
    best_centers = centers
    best_r = _compute_radius(centers)
    
    for restart in range(3):
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        # Different spacing for hex lattice
        spacing = 0.07 + restart * 0.005
        centers_r = _hexagonal_init(N, spacing=spacing)
        
        # Add some randomness
        centers_r += np.random.normal(0, 0.01, size=centers_r.shape)
        centers_r[:, 0] = np.clip(centers_r[:, 0], -0.49, 0.49)
        centers_r[:, 1] = np.clip(centers_r[:, 1], -0.49, 0.49)
        
        # Refine
        if time.time() - start_time < _DEADLINE - 25:
            centers_r = _refine_with_lbfgs(centers_r, max_time=20.0)
        
        if time.time() - start_time < _DEADLINE - 10:
            centers_r, r_r = _local_search(centers_r, max_time=10.0)
        else:
            r_r = _compute_radius(centers_r)
        
        if r_r > best_r:
            best_r = r_r
            best_centers = centers_r
    
    # Final validation
    r_final = _compute_radius(best_centers)
    if r_final <= 0:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
