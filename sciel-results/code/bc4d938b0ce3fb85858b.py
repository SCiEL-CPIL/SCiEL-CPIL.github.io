"""
Pack 200 equal circles in a unit square using Bottleneck Elimination 
Simulated Annealing on the exact non-smooth objective.
"""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
_TIME_LIMIT = 145  # Leave margin for 150s kill

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

def _find_bottlenecks(pts, r):
    """
    Find indices of pairs and walls that are active (limiting the radius).
    Returns:
        pair_indices: list of (i, j)
        wall_indices: list of (k, axis, side) where side 0 is min, 1 is max
    """
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    
    # Find pairs with distance close to 2*r
    # We want d_ij / 2 <= r + epsilon
    threshold = 2 * r + 1e-6
    pair_indices = []
    # Vectorized finding of indices
    # D is symmetric, we only need upper triangle
    triu_mask = np.triu(np.ones(D.shape, dtype=bool), k=1)
    close_pairs = np.argwhere(triu_mask & (D <= threshold))
    for i, j in close_pairs:
        pair_indices.append((int(i), int(j)))
        
    wall_indices = []
    # Walls: x_min, x_max, y_min, y_max
    # x_min: pts[:, 0] + 0.5 <= r + eps
    # x_max: 0.5 - pts[:, 0] <= r + eps
    # y_min: pts[:, 1] + 0.5 <= r + eps
    # y_max: 0.5 - pts[:, 1] <= r + eps
    
    eps = 1e-6
    x_min_idx = np.where(pts[:, 0] + 0.5 <= r + eps)[0]
    for k in x_min_idx:
        wall_indices.append((int(k), 0, 0))
        
    x_max_idx = np.where(0.5 - pts[:, 0] <= r + eps)[0]
    for k in x_max_idx:
        wall_indices.append((int(k), 0, 1))
        
    y_min_idx = np.where(pts[:, 1] + 0.5 <= r + eps)[0]
    for k in y_min_idx:
        wall_indices.append((int(k), 1, 0))
        
    y_max_idx = np.where(0.5 - pts[:, 1] <= r + eps)[0]
    for k in y_max_idx:
        wall_indices.append((int(k), 1, 1))
        
    return pair_indices, wall_indices

def _bottleneck_sa(pts, start_time, deadline, max_iter=5000, T_init=0.01, T_min=1e-6, alpha=0.95):
    """
    Simulated Annealing on the exact radius objective, focusing on bottlenecks.
    """
    pts = pts.copy()
    r = _compute_radius_fast(pts)
    best_r = r
    best_pts = pts.copy()
    
    T = T_init
    rng = np.random.default_rng()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Find current bottlenecks
        pair_indices, wall_indices = _find_bottlenecks(pts, r)
        
        if not pair_indices and not wall_indices:
            # No bottlenecks? Should not happen if r is computed correctly, 
            # unless r is limited by something else or numerical issues.
            # If no bottlenecks, we can't improve? 
            # Actually, if no constraints are active, we can move freely.
            # But r is min(d/2, w). If no d/2 <= r and no w <= r, then r < min(d/2, w).
            # This implies we can increase r.
            # In that case, we should just do a random move or a gradient step.
            # For simplicity, if no bottlenecks, pick a random pair and push apart.
            i, j = rng.integers(0, N, 2)
            if i == j: continue
            pair_indices = [(i, j)]
            
        # Choose a move
        # Bias towards pairs if they are the limiting factor, or walls.
        # For simplicity, pick randomly from all available bottlenecks.
        # If there are many, pick a random one.
        
        move_type = None
        if pair_indices and (not wall_indices or rng.random() < 0.7):
            move_type = 'pair'
            i, j = pair_indices[rng.integers(0, len(pair_indices))]
        else:
            move_type = 'wall'
            k, axis, side = wall_indices[rng.integers(0, len(wall_indices))]
            
        new_pts = pts.copy()
        
        # Generate move
        # Step size proportional to current radius
        step = r * 0.1
        
        if move_type == 'pair':
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1e-10:
                # Random direction
                angle = rng.uniform(0, 2*np.pi)
                dx, dy = np.cos(angle), np.sin(angle)
                dist = 1.0
            else:
                dx, dy = dx/dist, dy/dist
                
            # Move i away from j, j away from i
            # Add some jitter to escape local minima
            jitter_mag = step * 0.5 * rng.random()
            jitter_angle = rng.uniform(0, 2*np.pi)
            jx, jy = jitter_mag * np.cos(jitter_angle), jitter_mag * np.sin(jitter_angle)
            
            new_pts[i, 0] += dx * step + jx
            new_pts[i, 1] += dy * step + jy
            new_pts[j, 0] -= dx * step - jx
            new_pts[j, 1] -= dy * step - jy
            
            # Clip to square
            new_pts = np.clip(new_pts, -0.5, 0.5)
            
        else: # wall
            k, axis, side = move_type_data # Wait, I need to unpack this properly
            # Re-unpack from the chosen index
            k, axis, side = wall_indices[rng.integers(0, len(wall_indices))]
            
            # Move away from wall
            # axis 0: x, axis 1: y
            # side 0: min (-0.5), side 1: max (0.5)
            
            if axis == 0:
                if side == 0: # Left wall, move right (+x)
                    new_pts[k, 0] += step
                else: # Right wall, move left (-x)
                    new_pts[k, 0] -= step
            else:
                if side == 0: # Bottom wall, move up (+y)
                    new_pts[k, 1] += step
                else: # Top wall, move down (-y)
                    new_pts[k, 1] -= step
            
            # Add jitter perpendicular to wall
            jitter_mag = step * 0.5 * rng.random()
            if axis == 0:
                new_pts[k, 1] += jitter_mag * (1 if rng.random() < 0.5 else -1)
            else:
                new_pts[k, 0] += jitter_mag * (1 if rng.random() < 0.5 else -1)
            
            new_pts = np.clip(new_pts, -0.5, 0.5)
            
        # Evaluate
        new_r = _compute_radius_fast(new_pts)
        
        # Acceptance criterion
        if new_r > r:
            pts = new_pts
            r = new_r
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        else:
            # Simulated annealing acceptance
            delta = r - new_r # Positive if worse
            if delta < 1e-12:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
                
            if accept:
                pts = new_pts
                r = new_r
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_pts, best_r

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

def _local_polish(pts, max_iters=50, deadline=None):
    """
    Simple local polish: move the most constrained particle slightly.
    """
    if deadline is None:
        deadline = time.time() + 10
        
    pts = pts.copy()
    r = _compute_radius_fast(pts)
    
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Find the single most constraining factor
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_d = np.min(D)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        
        wall_dists = np.array([
            pts[:, 0] + 0.5,
            0.5 - pts[:, 0],
            pts[:, 1] + 0.5,
            0.5 - pts[:, 1]
        ])
        min_w = np.min(wall_dists)
        
        if min_d / 2.0 < min_w:
            # Pair is limiting
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1e-10:
                nx, ny = dx/dist, dy/dist
                move = 0.01 * r
                old_i = pts[i].copy()
                pts[i, 0] = np.clip(pts[i, 0] + nx * move, -0.5, 0.5)
                pts[i, 1] = np.clip(pts[i, 1] + ny * move, -0.5, 0.5)
                new_r = _compute_radius_fast(pts)
                if new_r <= r:
                    pts[i] = old_i
                else:
                    r = new_r
        else:
            # Wall is limiting
            min_idx = np.argmin(wall_dists)
            axis = min_idx // 2
            side = min_idx % 2
            k = min_idx % N # This logic is flawed, wall_dists is flattened
            
            # Re-calculate which point and wall
            # wall_dists is [x_min_all, x_max_all, y_min_all, y_max_all]
            # We need the index in the original array
            # Let's just re-find the min point
            if min_idx < N:
                k = min_idx; axis = 0; side = 0
            elif min_idx < 2*N:
                k = min_idx - N; axis = 0; side = 1
            elif min_idx < 3*N:
                k = min_idx - 2*N; axis = 1; side = 0
            else:
                k = min_idx - 3*N; axis = 1; side = 1
                
            move = 0.01 * r
            old_k = pts[k].copy()
            if axis == 0:
                if side == 0: pts[k, 0] += move
                else: pts[k, 0] -= move
            else:
                if side == 0: pts[k, 1] += move
                else: pts[k, 1] -= move
                
            pts = np.clip(pts, -0.5, 0.5)
            new_r = _compute_radius_fast(pts)
            if new_r <= r:
                pts[k] = old_k
            else:
                r = new_r
                
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
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
        if time.time() > deadline - 10:
            break
        
        try:
            coords = init_func()
            r = _compute_radius_fast(coords)
            
            if r > best_radius:
                best_radius = r
                best_coords = coords.copy()
            
            # Run Bottleneck SA
            # Allocate time based on remaining budget
            remaining = deadline - time.time() - 5
            if remaining > 10:
                # Max iterations roughly proportional to time
                # Estimate ~1ms per iter? No, cdist is 200x200, fast.
                # Let's say 1000 iters per second.
                max_iter = int(remaining * 1000)
                if max_iter > 20000: max_iter = 20000 # Cap it
                
                sa_coords, sa_r = _bottleneck_sa(
                    coords, 
                    start_time, 
                    deadline, 
                    max_iter=max_iter,
                    T_init=0.005,
                    T_min=1e-7
                )
                
                if sa_r > best_radius:
                    best_radius = sa_r
                    best_coords = sa_coords.copy()
                    
                # Polish
                if time.time() < deadline - 3:
                    polished = _local_polish(sa_coords, max_iters=20, deadline=deadline)
                    pol_r = _compute_radius_fast(polished)
                    if pol_r > best_radius:
                        best_radius = pol_r
                        best_coords = polished.copy()
                        
        except Exception as e:
            continue
    
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
