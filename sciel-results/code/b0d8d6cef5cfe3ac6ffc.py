"""Voronoi-based circle packing optimization for 50 circles in a unit square."""
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import minimize
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _get_delaunay_edges(centers):
    """Get unique edges from Delaunay triangulation."""
    try:
        tri = Delaunay(centers)
        simplices = tri.simplices
        edges = set()
        for simplex in simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _compute_voronoi_inradii(centers):
    """
    Compute the inradius of each Voronoi cell for the given centers.
    Returns an array of inradii (one per center).
    """
    try:
        vor = Voronoi(centers)
        inradii = np.zeros(N)
        
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                # Unbounded region - use wall distance
                inradii[i] = np.min(0.5 - np.abs(centers[i]))
                continue
            
            # Get the vertices of this Voronoi cell
            vertices = vor.vertices[region]
            
            if len(vertices) < 3:
                # Degenerate case
                inradii[i] = np.min(0.5 - np.abs(centers[i]))
                continue
            
            # Find the inscribed circle in this polygon
            # The inradius is the minimum distance from the center to any edge
            min_dist = np.inf
            
            for k in range(len(vertices)):
                v1 = vertices[k]
                v2 = vertices[(k+1) % len(vertices)]
                
                # Distance from center to line through v1 and v2
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                length = np.sqrt(dx*dx + dy*dy)
                
                if length < 1e-15:
                    continue
                    
                # Distance from point to line
                dist = np.abs(dy*(centers[i][0] - v1[0]) - dx*(centers[i][1] - v1[1])) / length
                min_dist = min(min_dist, dist)
            
            # Also consider wall constraints
            wall_dist = np.min(0.5 - np.abs(centers[i]))
            inradii[i] = min(min_dist, wall_dist)
        
        return inradii
    except Exception:
        # Fallback: use pairwise and wall distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        inradii = np.zeros(N)
        for i in range(N):
            min_pair = np.min(dists[i]) / 2.0
            wall_dist = np.min(0.5 - np.abs(centers[i]))
            inradii[i] = min(min_pair, wall_dist)
        return inradii

def _voronoi_objective(params, k):
    """
    Objective function that maximizes the minimum inradius of Voronoi cells.
    Uses soft-min for differentiability.
    """
    centers = params.reshape(N, 2)
    centers = np.clip(centers, -0.5, 0.5)
    
    inradii = _compute_voronoi_inradii(centers)
    
    if len(inradii) == 0:
        return 0.0
        
    m = np.min(inradii)
    shifted = inradii - m
    exp_terms = np.exp(-k * shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min = m - (1.0 / k) * log_sum
    return -soft_min

def _optimize_voronoi(start_centers, bounds, max_time=15.0):
    """
    Optimize using Voronoi-based objective with adaptive k.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    start_time = time.time()
    
    # Progressive k values for annealing
    k_schedule = [100.0, 300.0, 500.0, 1000.0, 2000.0]
    
    current_centers = best_centers.copy()
    
    for k in k_schedule:
        if time.time() - start_time > max_time:
            break
            
        try:
            result = minimize(
                _voronoi_objective,
                current_centers.reshape(-1),
                args=(k,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
            )
            
            centers = result.x.reshape(N, 2)
            centers = np.clip(centers, -0.5, 0.5)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
                current_centers = centers
        except Exception:
            continue
    
    return best_centers, best_score

def _generate_hex_init(spacing=0.14, seed=42):
    """Generate a hexagonal grid initialization."""
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

def _generate_square_init(seed=123):
    """Generate a square grid initialization."""
    np.random.seed(seed)
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        indices = np.random.choice(len(grid_points), N, replace=False)
        return grid_points[indices]
    return None

def _generate_random_init(seed=999):
    """Generate a random uniform initialization."""
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _generate_jittered_hex_init(spacing=0.14, seed=43):
    """Generate a jittered hexagonal initialization."""
    base = _generate_hex_init(spacing, seed)
    if base is None:
        return None
    np.random.seed(seed + 1)
    jittered = base + np.random.normal(0, 0.02, (N, 2))
    return np.clip(jittered, -0.49, 0.49)

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # Multiple hexagonal spacings
    for spacing in [0.13, 0.14, 0.15]:
        init = _generate_hex_init(spacing, seed=42 + int(spacing * 100))
        if init is not None:
            inits.append(init)
    
    # Square grid
    init = _generate_square_init(seed=123)
    if init is not None:
        inits.append(init)
    
    # Random inits with different seeds
    for seed in [999, 1000, 2000]:
        inits.append(_generate_random_init(seed))
    
    # Jittered hex
    init = _generate_jittered_hex_init(seed=43)
    if init is not None:
        inits.append(init)
    
    # Randomized grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])
    
    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    time_per_init = 18.0  # Leave time for safety margin
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 105.0:
            break
        
        centers, score = _optimize_voronoi(init, bounds, max_time=time_per_init)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
    
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
