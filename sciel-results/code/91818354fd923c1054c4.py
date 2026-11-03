"""
Pack 200 equal circles in a unit square using force-directed spreading
combined with soft-min gradient optimization and multi-start.
"""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

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

def _force_directed_step(coords, target_r, n_steps=50, dt_scale=0.1):
    """
    Force-directed repulsion using KDTree for efficient neighbor lookup.
    Points repel each other when closer than 2*target_r, and are pulled
    inward when too close to walls.
    """
    coords = coords.copy()
    n = len(coords)
    
    for step in range(n_steps):
        tree = cKDTree(coords)
        # Find all pairs within 2*target_r * 1.5
        pairs = tree.query_pairs(r=2 * target_r * 1.5)
        
        forces = np.zeros_like(coords)
        
        for i, j in pairs:
            d = coords[i] - coords[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            # Repulsive force: stronger when closer than 2*target_r
            if dist < 2 * target_r:
                force_mag = (2 * target_r - dist) / (2 * target_r)
                forces[i] += (d / dist) * force_mag
                forces[j] -= (d / dist) * force_mag
        
        # Wall forces: push inward if within target_r of wall
        for axis in range(2):
            # Left/bottom wall
            close_left = coords[:, axis] + 0.5 < target_r
            if np.any(close_left):
                forces[close_left, axis] += (target_r - (coords[close_left, axis] + 0.5)) / target_r
            # Right/top wall
            close_right = 0.5 - coords[:, axis] < target_r
            if np.any(close_right):
                forces[close_right, axis] -= (target_r - (0.5 - coords[close_right, axis])) / target_r
        
        # Apply forces with adaptive step size
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-10:
            break
        dt = dt_scale * target_r / max_force
        coords += dt * forces
        coords = np.clip(coords, -0.5, 0.5)
    
    return coords

def _soft_min_gradient(coords, softness=0.005):
    """
    Compute gradient of a soft-min approximation of the radius.
    Uses log-sum-exp for numerical stability.
    """
    n = len(coords)
    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)
    
    # Wall distances
    wall_dists = np.array([
        coords[:, 0] + 0.5,
        0.5 - coords[:, 0],
        coords[:, 1] + 0.5,
        0.5 - coords[:, 1]
    ]).T  # shape (n, 4)
    
    # All distances (pairwise + wall)
    # For each point, we have 4 wall distances and (n-1) pairwise distances
    # We'll compute soft-min over all constraints
    
    grad = np.zeros_like(coords)
    
    # Soft-min over pairwise distances
    # For each pair (i,j), the constraint is D[i,j] >= 2r
    # Soft-min: -log-sum-exp(-D/softness)
    # Gradient contribution to point i from pair (i,j):
    #   -exp(-D[i,j]/softness) / sum_k exp(-D[i,k]/softness) * (d D[i,j]/d c_i) / softness
    
    # This is expensive; use a simpler approach with top-k closest neighbors
    tree = cKDTree(coords)
    k = min(10, n - 1)
    dists, indices = tree.query(coords, k=k + 1)  # k+1 because includes self
    
    for i in range(n):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            d = dists[i, j_idx]
            if d < 1e-12:
                continue
            # Weight by soft-min
            weight = np.exp(-d / softness)
            direction = (coords[i] - coords[j]) / d
            grad[i] -= weight * direction / (2 * softness)
    
    # Wall gradients
    for axis in range(2):
        # Left/bottom
        d_wall = coords[:, axis] + 0.5
        w = np.exp(-d_wall / softness)
        grad[:, axis] -= w / softness
        # Right/top
        d_wall = 0.5 - coords[:, axis]
        w = np.exp(-d_wall / softness)
        grad[:, axis] += w / softness
    
    return grad

def _gradient_optimize(coords, target_r, max_iter=200, time_limit=None, start_time=None):
    """
    Gradient-based optimization to maximize soft-min radius.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = _TIME_LIMIT - 10
    
    coords = coords.copy()
    n = len(coords)
    lr = 0.01 * target_r
    softness = 0.003
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        grad = _soft_min_gradient(coords, softness)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        # Take step
        new_coords = coords + lr * grad
        new_coords = np.clip(new_coords, -0.5, 0.5)
        
        # Check if improved
        old_r = _compute_radius_fast(coords)
        new_r = _compute_radius_fast(new_coords)
        
        if new_r > old_r:
            coords = new_coords
            # Adaptive learning rate
            lr = min(lr * 1.1, 0.05 * target_r)
        else:
            # Halve learning rate
            lr *= 0.5
            if lr < 1e-8 * target_r:
                break
    
    return coords

def _local_optimize(coords, max_iter=500, time_limit=None, start_time=None):
    """
    Combined local optimizer: force-directed then gradient-based.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = _TIME_LIMIT - 10
    
    coords = coords.copy()
    n = len(coords)
    
    # Phase 1: Force-directed spreading
    current_r = _compute_radius_fast(coords)
    target_r = current_r * 1.1
    fd_time = time.time()
    fd_limit = min(30, time_limit - 15)
    if fd_limit > 5:
        coords = _force_directed_step(coords, target_r, n_steps=100, dt_scale=0.05)
        r_after_fd = _compute_radius_fast(coords)
        if r_after_fd > current_r:
            current_r = r_after_fd
    
    # Phase 2: Gradient-based refinement
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        coords = _gradient_optimize(coords, current_r, max_iter=300,
                                     time_limit=remaining, start_time=start_time)
        r_after_grad = _compute_radius_fast(coords)
        if r_after_grad > current_r:
            current_r = r_after_grad
    
    # Phase 3: Final targeted bottleneck moves
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        for it in range(100):
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
            
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > 1e-12:
                move = 0.02 * r
                nx, ny = dx / dist, dy / dist
                
                new_i = coords[i].copy()
                new_i[0] += nx * move
                new_i[1] += ny * move
                new_i = np.clip(new_i, -0.5, 0.5)
                
                coords[i] = new_i
                new_r = _compute_radius_fast(coords)
                if new_r <= r:
                    coords[i] = new_i - np.array([nx * move, ny * move])
                
                new_j = coords[j].copy()
                new_j[0] -= nx * move
                new_j[1] -= ny * move
                new_j = np.clip(new_j, -0.5, 0.5)
                
                coords[j] = new_j
                new_r2 = _compute_radius_fast(coords)
                if new_r2 <= r:
                    coords[j] = new_j + np.array([nx * move, ny * move])
            
            # Move wall-violating points inward
            for axis in range(2):
                for side in range(2):
                    if side == 0:
                        wall_dist = coords[:, axis] + 0.5
                    else:
                        wall_dist = 0.5 - coords[:, axis]
                    
                    min_idx = np.argmin(wall_dist)
                    if wall_dist[min_idx] < r:
                        if side == 0:
                            coords[min_idx, axis] += 0.02 * r
                        else:
                            coords[min_idx, axis] -= 0.02 * r
                        coords[min_idx, axis] = np.clip(coords[min_idx, axis], -0.5, 0.5)
    
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
        points += jitter * np.random.randn(n, 2)
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
        grid += jitter * np.random.randn(n, 2)
        grid = np.clip(grid, -0.5, 0.5)
    
    return grid

def _random_initialization(n):
    """
    Generate random points.
    """
    return np.random.uniform(-0.45, 0.45, (n, 2))

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Multiple diverse starts
    initializers = [
        ("hex", lambda: _hexagonal_initialization(N, jitter=0.0)),
        ("hex_jitter", lambda: _hexagonal_initialization(N, jitter=0.015)),
        ("grid", lambda: _grid_initialization(N, jitter=0.0)),
        ("grid_jitter", lambda: _grid_initialization(N, jitter=0.015)),
        ("random", lambda: _random_initialization(N)),
    ]
    
    best_coords = None
    best_radius = 0.0
    
    for name, init_func in initializers:
        if time.time() > deadline - 25:
            break
        
        try:
            coords = init_func()
            r = _compute_radius_fast(coords)
            
            if r > best_radius:
                best_radius = r
                best_coords = coords.copy()
            
            # Local optimization
            remaining_time = deadline - time.time() - 5
            if remaining_time > 15:
                optimized = _local_optimize(coords, max_iter=300, 
                                           time_limit=remaining_time,
                                           start_time=start_time)
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
