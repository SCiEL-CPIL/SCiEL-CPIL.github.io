"""
Pack 200 equal circles in a unit square using soft-min L-BFGS-B optimization
with force-directed diversification and multi-start annealing.
"""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 140  # Leave margin for 150s kill

def _compute_radius_fast(pts):
    """
    Fast radius computation using cdist.
    """
    if len(pts) == 0 or not np.all(np.isfinite(pts)):
        return 0.0
    
    # Wall distances
    min_wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    if min_wall < 0:
        return 0.0

    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min_objective_and_grad(x, beta):
    """
    Objective and gradient for soft-min maximization.
    We minimize -soft_min(constraints).
    
    x: flattened (2N,) array of point coordinates
    beta: sharpness parameter
    Returns: (objective_value, gradient)
    """
    pts = x.reshape((N, 2))
    
    # Wall constraint values (4N total)
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair constraint values: N*(N-1)/2
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    pair_dists = np.sqrt(dx**2 + dy**2)
    pair_vals = pair_dists / 2.0
    
    # All constraint values
    C = np.concatenate([wall_vals, pair_vals])
    
    # Soft min computation
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0 or not np.isfinite(sum_exp):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights for gradient
    weights = exp_shifted / sum_exp
    
    # Gradient computation
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]
    
    # Pair gradients using np.add.at for vectorized accumulation
    valid = pair_dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, pair_dists, 1.0)
    
    # d(dist/2)/dx_i = dx / (2 * dist), d(dist/2)/dx_j = -dx / (2 * dist)
    pair_weights = weights[4*N:] / (2.0 * dists_safe)
    pair_weights = np.where(valid, pair_weights, 0.0)
    
    # Accumulate gradients for point i (positive) and point j (negative)
    np.add.at(grad_pts[:, 0], i_idx, pair_weights * dx_safe)
    np.add.at(grad_pts[:, 0], j_idx, -pair_weights * dx_safe)
    np.add.at(grad_pts[:, 1], i_idx, pair_weights * dy_safe)
    np.add.at(grad_pts[:, 1], j_idx, -pair_weights * dy_safe)
    
    # Flatten gradient
    grad = grad_pts.flatten()
    
    return obj_val, grad

def _lbfgs_optimize(coords, beta, max_iter=300, time_limit=None, start_time=None):
    """
    Run L-BFGS-B optimization on the soft-min objective.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = _TIME_LIMIT - 10
    
    x0 = coords.flatten()
    
    def fun_and_grad(x):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            # Return current value to stop
            obj, grad = _soft_min_objective_and_grad(x, beta)
            return obj, grad
        return _soft_min_objective_and_grad(x, beta)
    
    result = minimize(
        fun_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': max_iter,
            'maxls': 50,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxfun': 500,
            'disp': False
        },
        bounds=[(-0.5, 0.5)] * (2 * N)
    )
    
    new_coords = result.x.reshape((N, 2))
    new_coords = np.clip(new_coords, -0.5, 0.5)
    
    return new_coords

def _force_directed_step(coords, target_r, n_steps=50, dt_scale=0.1):
    """
    Force-directed repulsion using KDTree for efficient neighbor lookup.
    """
    coords = coords.copy()
    n = len(coords)
    
    for step in range(n_steps):
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=2 * target_r * 1.5)
        
        forces = np.zeros_like(coords)
        
        for i, j in pairs:
            d = coords[i] - coords[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            if dist < 2 * target_r:
                force_mag = (2 * target_r - dist) / (2 * target_r)
                forces[i] += (d / dist) * force_mag
                forces[j] -= (d / dist) * force_mag
        
        for axis in range(2):
            close_left = coords[:, axis] + 0.5 < target_r
            if np.any(close_left):
                forces[close_left, axis] += (target_r - (coords[close_left, axis] + 0.5)) / target_r
            close_right = 0.5 - coords[:, axis] < target_r
            if np.any(close_right):
                forces[close_right, axis] -= (target_r - (0.5 - coords[close_right, axis])) / target_r
        
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-10:
            break
        dt = dt_scale * target_r / max_force
        coords += dt * forces
        coords = np.clip(coords, -0.5, 0.5)
    
    return coords

def _exact_local_polish(coords, max_iters=200, time_limit=None, start_time=None):
    """
    Final high-precision polish: iteratively move the bottleneck pair apart
    and fix wall violations with small exact steps.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = _TIME_LIMIT - 5
    
    coords = coords.copy()
    n = len(coords)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        D = cdist(coords, coords)
        np.fill_diagonal(D, np.inf)
        min_dist = np.min(D)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        
        wall_dists = np.array([
            coords[:, 0] + 0.5,
            0.5 - coords[:, 0],
            coords[:, 1] + 0.5,
            0.5 - coords[:, 1]
        ])
        min_wall = np.min(wall_dists)
        
        r = min(min_dist / 2.0, min_wall)
        if r <= 0:
            break
        
        # Try to separate the closest pair
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 1e-12:
            nx, ny = dx / dist, dy / dist
            move = 0.01 * r
            
            # Try moving i away from j
            old_i = coords[i].copy()
            coords[i, 0] = np.clip(coords[i, 0] + nx * move, -0.5, 0.5)
            coords[i, 1] = np.clip(coords[i, 1] + ny * move, -0.5, 0.5)
            new_r = _compute_radius_fast(coords)
            if new_r <= r:
                coords[i] = old_i
            
            # Try moving j away from i
            old_j = coords[j].copy()
            coords[j, 0] = np.clip(coords[j, 0] - nx * move, -0.5, 0.5)
            coords[j, 1] = np.clip(coords[j, 1] - ny * move, -0.5, 0.5)
            new_r = _compute_radius_fast(coords)
            if new_r <= r:
                coords[j] = old_j
        
        # Fix wall violations
        for axis in range(2):
            for side in range(2):
                if side == 0:
                    wall_dist = coords[:, axis] + 0.5
                    min_idx = np.argmin(wall_dist)
                    if wall_dist[min_idx] < r:
                        old_val = coords[min_idx, axis].copy()
                        coords[min_idx, axis] = np.clip(coords[min_idx, axis] + 0.01 * r, -0.5, 0.5)
                        if _compute_radius_fast(coords) < r:
                            coords[min_idx, axis] = old_val
                else:
                    wall_dist = 0.5 - coords[:, axis]
                    min_idx = np.argmin(wall_dist)
                    if wall_dist[min_idx] < r:
                        old_val = coords[min_idx, axis].copy()
                        coords[min_idx, axis] = np.clip(coords[min_idx, axis] - 0.01 * r, -0.5, 0.5)
                        if _compute_radius_fast(coords) < r:
                            coords[min_idx, axis] = old_val
    
    return coords

def _hexagonal_initialization(n, jitter=0.0):
    """
    Generate a hexagonal layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m
    
    cols = int(1.0 / (2 * r_est)) + 2
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2
    
    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])
            
    points = np.array(points)
    
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) < n:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:n]
    else:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)
        points = points[indices[:n]]
    
    if jitter > 0:
        points = points + jitter * np.random.randn(n, 2)
        points = np.clip(points, -0.5, 0.5)
    
    return points

def _grid_initialization(n, jitter=0.0):
    """
    Generate a grid layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) < n:
        extra = n - len(grid)
        extra_pts = np.random.uniform(-0.45, 0.45, (extra, 2))
        grid = np.vstack([grid, extra_pts])
    
    grid = grid[:n]
    
    if jitter > 0:
        grid = grid + jitter * np.random.randn(n, 2)
        grid = np.clip(grid, -0.5, 0.5)
    
    return grid

def _random_initialization(n):
    """
    Generate random points.
    """
    return np.random.uniform(-0.45, 0.45, (n, 2))

def _optimize_single_start(coords, start_time, deadline):
    """
    Full optimization pipeline for a single starting configuration:
    1. Force-directed spreading to get a reasonable initial spread
    2. Annealed soft-min L-BFGS-B optimization (decreasing beta)
    3. Exact local polish
    """
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return coords
    
    current_r = _compute_radius_fast(coords)
    
    # Phase 1: Force-directed spreading
    if time.time() - start_time < deadline - 25:
        target_r = current_r * 1.2
        coords_fd = _force_directed_step(coords, target_r, n_steps=80, dt_scale=0.05)
        r_fd = _compute_radius_fast(coords_fd)
        if r_fd > current_r:
            coords = coords_fd
            current_r = r_fd
    
    # Phase 2: Annealed L-BFGS-B optimization
    # Start with soft beta, anneal to sharp beta
    beta_schedule = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    
    for beta in beta_schedule:
        if time.time() - start_time > deadline - 15:
            break
        
        remaining = deadline - time.time() - 10
        if remaining < 5:
            break
        
        # Estimate max iterations based on remaining time
        # Each L-BFGS iteration takes roughly 5-10ms for N=200
        max_iter = min(200, int(remaining * 100))
        if max_iter < 20:
            break
        
        try:
            new_coords = _lbfgs_optimize(coords, beta, max_iter=max_iter,
                                         time_limit=remaining,
                                         start_time=start_time)
            new_r = _compute_radius_fast(new_coords)
            if new_r > current_r:
                coords = new_coords
                current_r = new_r
        except Exception:
            continue
    
    # Phase 3: Exact local polish
    if time.time() - start_time < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 3:
            coords_polished = _exact_local_polish(coords, max_iters=100,
                                                   time_limit=remaining,
                                                   start_time=start_time)
            r_polished = _compute_radius_fast(coords_polished)
            if r_polished > current_r:
                coords = coords_polished
                current_r = r_polished
    
    return coords

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Multiple diverse starts
    initializers = [
        ("hex", lambda: _hexagonal_initialization(N, jitter=0.0)),
        ("hex_jitter", lambda: _hexagonal_initialization(N, jitter=0.01)),
        ("grid", lambda: _grid_initialization(N, jitter=0.0)),
        ("grid_jitter", lambda: _grid_initialization(N, jitter=0.01)),
        ("random", lambda: _random_initialization(N)),
    ]
    
    best_coords = None
    best_radius = 0.0
    
    for name, init_func in initializers:
        if time.time() > deadline - 20:
            break
        
        try:
            coords = init_func()
            r = _compute_radius_fast(coords)
            
            if r > best_radius:
                best_radius = r
                best_coords = coords.copy()
            
            # Full optimization pipeline
            remaining = deadline - time.time() - 10
            if remaining > 15:
                optimized = _optimize_single_start(coords, start_time, deadline)
                r_opt = _compute_radius_fast(optimized)
                
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_coords = optimized.copy()
        except Exception:
            continue
    
    # Final validation
    if best_coords is None or not np.all(np.isfinite(best_coords)):
        return _grid_initialization(N)
    
    return best_coords

# EVOLVE-BLOCK-START
def compute_radius(coords):
    return _compute_radius_fast(coords)

def objective(coords):
    r = compute_radius(coords)
    return -r

def initial_hex_layout(n):
    return _hexagonal_initialization(n)

# EVOLVE-BLOCK-END
