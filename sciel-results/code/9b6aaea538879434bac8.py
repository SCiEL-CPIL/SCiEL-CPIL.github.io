"""Voronoi-based packing optimization for 50 equal circles in a unit square."""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time

N = 50

def _true_radius(centers):
    """Compute the true maximum common radius."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _voronoi_inradii(centers):
    """
    Compute the inradius of each Voronoi cell (distance from center to nearest edge/wall).
    Returns array of shape (N,) with the inradius for each cell.
    """
    # Build Voronoi diagram
    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback: just use wall distances
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        return wall_dists
    
    inradii = np.zeros(N)
    
    for i in range(N):
        # Get the Voronoi region for point i
        region = vor.regions[vor.point_region[i]]
        
        if len(region) == 0 or -1 in region:
            # Unbounded region: use distance to nearest wall
            wall_dists = 0.5 - np.abs(centers[i])
            inradii[i] = np.min(wall_dists)
            continue
        
        # Compute distance from center to each edge of the Voronoi cell
        min_dist = np.inf
        for j in range(len(region) - 1):
            v1 = vor.vertices[region[j]]
            v2 = vor.vertices[region[j + 1]]
            
            # Distance from point to line segment
            p = centers[i]
            a = v2 - v1
            b = v1 - p
            if np.dot(a, a) < 1e-12:
                d = np.linalg.norm(b)
            else:
                t = np.dot(b, a) / np.dot(a, a)
                t = np.clip(t, 0, 1)
                d = np.linalg.norm(b + t * a)
            min_dist = min(min_dist, d)
        
        # Also consider wall distances (Voronoi may extend beyond square)
        wall_dists = 0.5 - np.abs(centers[i])
        min_dist = min(min_dist, np.min(wall_dists))
        
        inradii[i] = min_dist
    
    return inradii

def _smooth_min(values, k=100.0):
    """Soft minimum of an array of values."""
    m = np.min(values)
    shifted = values - m
    # Use a stable log-sum-exp
    max_shifted = np.max(shifted)
    if max_shifted > 50:
        shifted = shifted - max_shifted
        log_sum = np.log(np.sum(np.exp(-k * shifted))) + max_shifted * k
    else:
        log_sum = np.log(np.sum(np.exp(-k * shifted)))
    return m - (1.0 / k) * log_sum

def _optimize_voronoi(centers, bounds, max_iter=3000):
    """
    Optimize circle positions using Voronoi cell inradii as the objective.
    Maximize the minimum inradius across all cells.
    """
    best_centers = centers.copy()
    best_score = _true_radius(best_centers)
    
    def objective(params, k):
        c = params.reshape(N, 2)
        c = np.clip(c, -0.5, 0.5)
        
        inradii = _voronoi_inradii(c)
        if len(inradii) == 0:
            return 0.0
        
        # We want to maximize the minimum inradius
        # Use soft min for differentiability
        soft_min_val = _smooth_min(inradii, k)
        return -soft_min_val
    
    # Phase 1: Coarse optimization
    try:
        result = minimize(
            objective,
            best_centers.reshape(-1),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        c = result.x.reshape(N, 2)
        c = np.clip(c, -0.5, 0.5)
        r = _true_radius(c)
        if r > best_score:
            best_score = r
            best_centers = c
    except Exception:
        pass
    
    # Phase 2: Fine optimization with higher k
    try:
        result = minimize(
            objective,
            best_centers.reshape(-1),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
        )
        c = result.x.reshape(N, 2)
        c = np.clip(c, -0.5, 0.5)
        r = _true_radius(c)
        if r > best_score:
            best_score = r
            best_centers = c
    except Exception:
        pass
    
    # Phase 3: Final polish with very high k
    try:
        result = minimize(
            objective,
            best_centers.reshape(-1),
            args=(500.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
        )
        c = result.x.reshape(N, 2)
        c = np.clip(c, -0.5, 0.5)
        r = _true_radius(c)
        if r > best_score:
            best_score = r
            best_centers = c
    except Exception:
        pass
    
    return best_centers, best_score

def _make_hex_init(seed=42, spacing=0.14):
    """Create a hexagonal grid initialization."""
    np.random.seed(seed)
    points = []
    range_x = np.arange(-0.8, 0.8, spacing)
    range_y = np.arange(-0.8, 0.8, spacing)
    for i, x in enumerate(range_x):
        offset = (spacing / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        indices = np.random.choice(len(points), N, replace=False)
        return points[indices]
    return None

def _make_square_init(seed=123, n_per_side=7):
    """Create a square grid initialization."""
    np.random.seed(seed)
    grid_x = np.linspace(-0.45, 0.45, n_per_side)
    grid_y = np.linspace(-0.45, 0.45, n_per_side)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        indices = np.random.choice(len(grid_points), N, replace=False)
        return grid_points[indices]
    return None

def _make_random_init(seed=999):
    """Create a random uniform initialization."""
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _make_jittered_hex_init(seed=43, spacing=0.14, jitter=0.02):
    """Create a jittered hexagonal grid initialization."""
    base = _make_hex_init(seed=seed, spacing=spacing)
    if base is None:
        return None
    np.random.seed(seed + 1)
    jittered = base + np.random.normal(0, jitter, (N, 2))
    jittered = np.clip(jittered, -0.49, 0.49)
    return jittered

def _make_random_grid_init(seed=555, n_per_side=7):
    """Create a randomized grid initialization."""
    np.random.seed(seed)
    grid_x = np.sort(np.random.uniform(-0.49, 0.49, n_per_side))
    grid_y = np.sort(np.random.uniform(-0.49, 0.49, n_per_side))
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        indices = np.random.choice(len(grid_points), N, replace=False)
        return grid_points[indices]
    return None

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid
    hex1 = _make_hex_init(seed=42, spacing=0.14)
    if hex1 is not None:
        inits.append(hex1)
    
    # 2. Square grid
    square1 = _make_square_init(seed=123, n_per_side=7)
    if square1 is not None:
        inits.append(square1)
    
    # 3. Random uniform
    rand1 = _make_random_init(seed=999)
    inits.append(rand1)
    
    # 4. Jittered hex
    jitter1 = _make_jittered_hex_init(seed=43, spacing=0.14, jitter=0.02)
    if jitter1 is not None:
        inits.append(jitter1)
    
    # 5. Another random
    rand2 = _make_random_init(seed=1000)
    inits.append(rand2)
    
    # 6. Randomized grid
    rand_grid1 = _make_random_grid_init(seed=555, n_per_side=7)
    if rand_grid1 is not None:
        inits.append(rand_grid1)
    
    # 7. Additional diverse inits
    hex2 = _make_hex_init(seed=77, spacing=0.13)
    if hex2 is not None:
        inits.append(hex2)
    
    square2 = _make_square_init(seed=246, n_per_side=8)
    if square2 is not None:
        inits.append(square2)
    
    jitter2 = _make_jittered_hex_init(seed=88, spacing=0.15, jitter=0.03)
    if jitter2 is not None:
        inits.append(jitter2)
    
    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 100.0:
            break
        
        # Use fewer iterations for more inits, more for fewer
        remaining_inits = len(inits) - i
        if remaining_inits > 5:
            max_iter = 1500
        elif remaining_inits > 3:
            max_iter = 2000
        else:
            max_iter = 3000
        
        centers, score = _optimize_voronoi(init, bounds, max_iter=max_iter)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    if best_centers is None:
        # Fallback
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
    
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
