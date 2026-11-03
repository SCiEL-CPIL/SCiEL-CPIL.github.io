"""Pack 200 equal circles in a unit square using Lennard-Jones potential + BFGS."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    x = centers[:, 0]
    y = centers[:, 1]
    min_wall = np.min(np.concatenate([
        0.5 - np.abs(x),
        0.5 - np.abs(y)
    ]))
    
    return min(min_pair, min_wall)


def _lj_potential(centers, eps=1.0, sigma=0.037, wall_k=50.0):
    """
    Lennard-Jones potential between pairs + wall confinement.
    V = sum_{i<j} eps*((sigma/d)^12 - 2*(sigma/d)^6) + wall_k * sum_i (penalty_i)
    
    The factor of 2 in the LJ term makes the minimum at d=sigma with V=0.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    # LJ potential: eps*((sigma/d)^12 - 2*(sigma/d)^6)
    # = eps * sigma^12/d^12 - 2*eps*sigma^6/d^6
    # Let's compute (sigma^2/d^2)^6 and (sigma^2/d^2)^3 for efficiency
    s2_over_d2 = (sigma * sigma) / dist_sq
    s6 = s2_over_d2 ** 3
    s12 = s6 * s6
    
    lj_matrix = eps * (s12 - 2.0 * s6)
    np.fill_diagonal(lj_matrix, 0.0)
    pair_energy = 0.5 * np.sum(lj_matrix)
    
    # Wall confinement: quadratic penalty for being close to walls
    x = centers[:, 0]
    y = centers[:, 1]
    # Distance to each wall
    dx_right = 0.5 - x
    dx_left = 0.5 + x
    dy_top = 0.5 - y
    dy_bot = 0.5 + y
    
    # Penalty: large when close to wall
    # Use a smooth penalty that grows as circle approaches wall
    wall_penalty = 0.0
    for d in [dx_right, dx_left, dy_top, dy_bot]:
        # Penalty when d < sigma/2 (circle touching wall)
        # Use quadratic: wall_k * max(0, sigma/2 - d)^2
        violation = np.maximum(0.0, sigma / 2.0 - d)
        wall_penalty += np.sum(violation ** 2)
    
    wall_energy = wall_k * wall_penalty
    
    return pair_energy + wall_energy


def _lj_gradient(centers, eps=1.0, sigma=0.037, wall_k=50.0):
    """Gradient of the LJ potential + wall confinement."""
    n = len(centers)
    grad = np.zeros_like(centers)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    # dV/d(centers_i) for pair (i,j):
    # V_ij = eps*((sigma/d)^12 - 2*(sigma/d)^6)
    # dV_ij/dd = eps*(-12*sigma^12/d^13 + 12*sigma^6/d^7)
    #         = 12*eps/d * (sigma^6/d^6 - sigma^12/d^12)
    #         = 12*eps/d * (s6 - s12) where s6=(sigma/d)^6, s12=(sigma/d)^12
    # dV_ij/d(centers_i) = dV_ij/dd * dd/d(centers_i) = dV_ij/dd * (centers_i - centers_j)/d
    #                    = 12*eps*(s6 - s12) * (centers_i - centers_j)/d^2
    
    s2_over_d2 = (sigma * sigma) / dist_sq
    s6 = s2_over_d2 ** 3
    s12 = s6 * s6
    
    # Force matrix: F_ij = 12*eps*(s6 - s12) / d^2 * (centers_i - centers_j)
    # Note: (s6 - s12) can be positive or negative
    coeff = 12.0 * eps * (s6 - s12) / dist_sq
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff[i,j] * (centers_i - centers_j)
    # = sum_j coeff[i,j] * centers_i - sum_j coeff[i,j] * centers_j
    # = (sum_j coeff[i,j]) * centers_i - sum_j coeff[i,j] * centers_j
    
    row_sum = np.sum(coeff, axis=1)
    grad = row_sum[:, np.newaxis] * centers - coeff @ centers
    
    # Wall gradient
    x = centers[:, 0]
    y = centers[:, 1]
    dx_right = 0.5 - x
    dx_left = 0.5 + x
    dy_top = 0.5 - y
    dy_bot = 0.5 + y
    
    for i in range(n):
        # d/dx of wall_penalty for right wall: -2*wall_k*(sigma/2 - dx_right) if dx_right < sigma/2
        v = np.maximum(0.0, sigma / 2.0 - dx_right[i])
        if v > 0:
            grad[i, 0] += 2.0 * wall_k * v  # d/dx of (sigma/2 - (0.5-x))^2 = -2*(sigma/2-(0.5-x))*(-1) = 2*(sigma/2-(0.5-x))
        
        v = np.maximum(0.0, sigma / 2.0 - dx_left[i])
        if v > 0:
            grad[i, 0] -= 2.0 * wall_k * v
        
        v = np.maximum(0.0, sigma / 2.0 - dy_top[i])
        if v > 0:
            grad[i, 1] += 2.0 * wall_k * v
        
        v = np.maximum(0.0, sigma / 2.0 - dy_bot[i])
        if v > 0:
            grad[i, 1] -= 2.0 * wall_k * v
    
    return grad


def _hex_init(spacing=0.04):
    """Hexagonal lattice initial positions."""
    points = []
    row = 0
    while len(points) < N:
        offset = (row % 2) * spacing * 0.5
        col = 0
        while len(points) < N:
            x = -0.5 + spacing / 2 + col * spacing + offset
            y = -0.5 + spacing * np.sqrt(3) / 2 * row
            if -0.499 <= x <= 0.499 and -0.499 <= y <= 0.499:
                points.append([x, y])
            col += 1
        row += 1
    return np.array(points[:N])


def _grid_init():
    """Grid initial positions."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init(seed=42):
    """Random initial positions inside the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _bottleneck_polish(centers, max_iter=500, time_limit=10.0):
    """
    Final polish: iteratively push the closest pair apart and fix wall violations.
    This operates on the exact non-smooth objective.
    """
    start_time = time.time()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find closest pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_dist = dist[i, j]
        
        # Check wall distances
        x = centers[:, 0]
        y = centers[:, 1]
        wall_dists = np.stack([
            0.5 - x,
            0.5 + x,
            0.5 - y,
            0.5 + y
        ], axis=1)
        min_wall = np.min(wall_dists)
        
        # If wall is the bottleneck, fix it
        if min_wall < min_dist / 2.0:
            # Find circle with minimum wall distance
            flat_idx = np.argmin(wall_dists)
            ci = flat_idx // 4
            wall_id = flat_idx % 4
            
            # Push away from wall
            if wall_id == 0:  # right wall
                centers[ci, 0] -= 0.001
            elif wall_id == 1:  # left wall
                centers[ci, 0] += 0.001
            elif wall_id == 2:  # top wall
                centers[ci, 1] -= 0.001
            else:  # bottom wall
                centers[ci, 1] += 0.001
        else:
            # Push closest pair apart
            direction = centers[i] - centers[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            direction = direction / norm
            
            # Move both apart by a small amount
            step = 0.0005
            centers[i] += direction * step
            centers[j] -= direction * step
        
        # Clip to valid range
        centers[:, 0] = np.clip(centers[:, 0], -0.499, 0.499)
        centers[:, 1] = np.clip(centers[:, 1], -0.499, 0.499)
        
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    return best_centers


def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s limit
    
    best_centers = _hex_init(0.04)
    best_radius = _compute_radius(best_centers)
    
    # Try multiple initializations with LJ+BFGS
    inits = [
        ("hex_004", _hex_init(0.04)),
        ("hex_0035", _hex_init(0.035)),
        ("hex_0045", _hex_init(0.045)),
        ("grid", _grid_init()),
        ("rand_42", _random_init(42)),
        ("rand_123", _random_init(123)),
    ]
    
    # LJ parameters
    eps_values = [1.0, 2.0, 5.0]
    sigma_values = [0.036, 0.037, 0.038]
    wall_k = 50.0
    
    for init_name, init in inits:
        if time.time() - start_time > deadline - 5:
            break
        
        for eps in eps_values:
            if time.time() - start_time > deadline - 5:
                break
            
            for sigma in sigma_values:
                if time.time() - start_time > deadline - 5:
                    break
                
                try:
                    result = minimize(
                        _lj_potential,
                        init.copy(),
                        jac=_lj_gradient,
                        args=(eps, sigma, wall_k),
                        method='L-BFGS-B',
                        bounds=[(-0.499, 0.499)] * (2 * N),
                        options={
                            'maxiter': 300,
                            'ftol': 1e-14,
                            'gtol': 1e-10,
                            'maxls': 30
                        }
                    )
                    
                    centers = result.x
                    centers[:, 0] = np.clip(centers[:, 0], -0.499, 0.499)
                    centers[:, 1] = np.clip(centers[:, 1], -0.499, 0.499)
                    
                    radius = _compute_radius(centers)
                    if radius > best_radius:
                        best_radius = radius
                        best_centers = centers.copy()
                        
                except Exception:
                    pass
    
    # Final bottleneck polish
    if time.time() - start_time < deadline - 3:
        polished = _bottleneck_polish(best_centers.copy(), max_iter=1000, time_limit=deadline - time.time() - 2)
        polished_radius = _compute_radius(polished)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished
    
    return best_centers

# EVOLVE-BLOCK-END
