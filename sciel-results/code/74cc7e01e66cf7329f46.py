"""Circle packing: 200 equal circles in unit square, maximize radius."""
import numpy as np
import time

N = 200
DEADLINE = 115.0  # safety margin before 120s limit


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Distance between all pairs
    n = len(centers)
    # Use broadcasting for pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to walls
    # Square is [-0.5, 0.5]^2
    wall_dist = np.min([
        0.5 - np.min(centers[:, 0]),
        0.5 + np.min(centers[:, 0]),
        0.5 - np.max(centers[:, 1]),
        0.5 + np.min(centers[:, 1]),
    ])
    # Actually need per-point distance to nearest wall
    # For each point, distance to each of 4 walls
    dx = np.minimum(centers[:, 0] + 0.5, 0.5 - centers[:, 0])
    dy = np.minimum(centers[:, 1] + 0.5, 0.5 - centers[:, 1])
    wall_dist = np.min(np.minimum(dx, dy))
    
    r = 0.5 * min(min_pair_dist, wall_dist)
    return r


def _hexagonal_seed(n, rng):
    """Generate initial positions on a hexagonal lattice pattern, then jitter."""
    # Hexagonal packing has density pi/(2*sqrt(3)) ≈ 0.9069
    # For n circles in unit square, optimal spacing ~ sqrt(1/n / density)
    # Let's create a hex grid and pick n points
    
    # Hexagonal lattice: rows offset by half spacing
    # Spacing between centers in hex packing: d = 2*r
    # Area per circle in hex: d^2 * sqrt(3)/2
    # Total area = n * d^2 * sqrt(3)/2 <= 1
    # d <= sqrt(2/(n*sqrt(3)))
    
    d = np.sqrt(2.0 / (n * np.sqrt(3.0)))
    row_spacing = d * np.sqrt(3.0) / 2.0
    
    # Generate hex grid points
    points = []
    y = -0.5
    row = 0
    while y <= 0.5:
        x_offset = (d / 2.0) if (row % 2 == 1) else 0.0
        x = -0.5 + x_offset
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            x += d
        y += row_spacing
        row += 1
    
    points = np.array(points)
    
    if len(points) >= n:
        # Randomly select n points
        idx = rng.choice(len(points), size=n, replace=False)
        centers = points[idx].copy()
    else:
        # Use all and add random points
        centers = points.copy()
        remaining = n - len(centers)
        extra = rng.uniform(-0.5, 0.5, size=(remaining, 2))
        centers = np.vstack([centers, extra])
    
    # Add small jitter to break symmetry
    centers += rng.normal(0, 0.01, size=centers.shape)
    # Clamp to square
    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    return centers


def _repulsion_relax(centers, n_iter=500, lr=0.001, rng=None):
    """
    Repulsive force-based relaxation to maximize minimum distance.
    Uses gradient ascent on soft-min of pairwise distances.
    """
    centers = centers.copy()
    n = len(centers)
    
    # Soft-min parameter: lower = more focused on minimum
    # We use log-sum-exp style soft-min for distances
    beta = 20.0  # larger = sharper min
    
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(n_iter):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        # Soft-min of distances: -log(sum(exp(-beta*dist)))/beta
        # Gradient of soft-min w.r.t. positions
        # d/dx_i softmin = sum_j (exp(-beta*d_ij) / sum_k exp(-beta*d_ik)) * (-d/dx_i d_ij)
        
        # Compute weights
        neg_beta_dist = -beta * dist
        # For numerical stability, subtract max
        neg_beta_dist_max = np.max(neg_beta_dist, axis=1, keepdims=True)
        exp_vals = np.exp(neg_beta_dist - neg_beta_dist_max)
        np.fill_diagonal(exp_vals, 0.0)
        sums = np.sum(exp_vals, axis=1, keepdims=True)
        weights = exp_vals / sums  # weight[i,j] = probability mass on pair (i,j)
        
        # Gradient of distance d_ij w.r.t. position i:
        # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
        # We want to maximize soft-min, so we move in direction of gradient
        # grad_i = sum_j weights[i,j] * (x_i - x_j) / d_ij
        
        # Compute unit vectors
        dist_safe = np.where(dist > 1e-10, dist, 1.0)
        unit_vec = diff / dist_safe[:, :, np.newaxis]
        np.fill_diagonal(unit_vec[:, :, 0], 0.0)
        np.fill_diagonal(unit_vec[:, :, 1], 0.0)
        
        # Gradient: sum_j weights[i,j] * unit_vec[i,j]
        grad = np.sum(weights[:, :, np.newaxis] * unit_vec, axis=1)
        
        # Wall constraints: push away from walls
        # For each point, if close to wall, push inward
        margin = 0.05
        # Left wall: x < -0.5 + margin
        left_close = centers[:, 0] < -0.5 + margin
        right_close = centers[:, 0] > 0.5 - margin
        bottom_close = centers[:, 1] < -0.5 + margin
        top_close = centers[:, 1] > 0.5 - margin
        
        wall_grad = np.zeros_like(centers)
        wall_grad[left_close, 0] += 1.0
        wall_grad[right_close, 0] -= 1.0
        wall_grad[bottom_close, 1] += 1.0
        wall_grad[top_close, 1] -= 1.0
        
        # Combine gradients
        total_grad = grad + 0.5 * wall_grad
        
        # Update
        centers += lr * total_grad
        
        # Clamp to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Adaptive learning rate: decrease over time
        lr = max(0.0001, 0.001 * (1 - it / n_iter) ** 2)
        
        # Check improvement
        if it % 50 == 0:
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
    
    return best_centers


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_r = -1.0
    
    # Multiple restarts with different seeds
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        if restart == 0:
            centers = _hexagonal_seed(N, rng)
        else:
            # Perturb the best solution found so far
            if best_centers is not None:
                centers = best_centers.copy()
                noise_scale = 0.02 * (restart + 1)
                centers += rng.normal(0, noise_scale, size=centers.shape)
                centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
                centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            else:
                centers = _hexagonal_seed(N, rng)
        
        # Relax
        centers = _repulsion_relax(centers, n_iter=300, lr=0.002, rng=rng)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Final polish: more iterations with smaller learning rate
    if best_centers is not None and time.time() - start_time < DEADLINE - 10:
        best_centers = _repulsion_relax(best_centers, n_iter=200, lr=0.0005, rng=rng)
        r = _compute_radius(best_centers)
        if r > best_r:
            best_r = r
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
