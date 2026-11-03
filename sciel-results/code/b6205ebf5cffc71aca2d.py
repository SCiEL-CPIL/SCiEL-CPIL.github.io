"""Voronoi-based sparse optimization for circle packing."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay
import time

N = 50

def _compute_voronoi_inradii(centers):
    """Compute the inradius of each Voronoi cell (distance from center to nearest edge)."""
    try:
        vor = Voronoi(centers)
        inradii = np.zeros(N)
        for i, point in enumerate(centers):
            # Get the Voronoi region for this point
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0:
                # Unbounded region - use distance to nearest other point / 2
                dists = np.linalg.norm(centers - point, axis=1)
                dists[i] = np.inf
                inradii[i] = np.min(dists) / 2.0
                # Also consider walls
                wall_dist = 0.5 - np.abs(point)
                inradii[i] = min(inradii[i], np.min(wall_dist))
                continue
            
            # For bounded regions, compute distance from point to each edge
            min_dist = np.inf
            for j in region:
                if j < 0:
                    continue
                vertex1 = vor.vertices[region[j]]
                vertex2 = vor.vertices[region[(j + 1) % len(region)]]
                # Distance from point to line segment vertex1-vertex2
                v = vertex2 - vertex1
                u = point - vertex1
                if np.dot(v, v) > 1e-15:
                    t = np.dot(u, v) / np.dot(v, v)
                    t = np.clip(t, 0, 1)
                    closest = vertex1 + t * v
                    dist = np.linalg.norm(point - closest)
                else:
                    dist = np.linalg.norm(point - vertex1)
                min_dist = min(min_dist, dist)
            
            # Also consider wall distances
            wall_dist = 0.5 - np.abs(point)
            inradii[i] = min(min_dist, np.min(wall_dist))
        
        return inradii
    except Exception:
        # Fallback: compute true radius
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        return np.full(N, min(min_pair, min_wall))

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

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _optimize_from_start(start_centers, bounds):
    """Optimization using Delaunay-aware sparse objective."""
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    def objective(params, k):
        centers = params.reshape(N, 2)
        centers = np.clip(centers, -0.5, 0.5)
        
        edges = _get_delaunay_edges(centers)
        
        active_dists = []
        for i, j in edges:
            d = np.linalg.norm(centers[i] - centers[j])
            active_dists.append(d / 2.0)
        
        wall_dists = 0.5 - np.abs(centers)
        active_dists.extend(wall_dists.flatten())
        
        active_dists = np.array(active_dists)
        if len(active_dists) == 0:
            return 0.0
            
        m = np.min(active_dists)
        shifted = active_dists - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min

    # Multiple optimization passes with increasing k
    for k_val in [100.0, 500.0, 2000.0, 5000.0]:
        try:
            result = minimize(
                objective, 
                best_centers.reshape(-1), 
                args=(k_val,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
            )
            centers = result.x.reshape(N, 2)
            centers = np.clip(centers, -0.5, 0.5)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
        except Exception:
            pass
    
    return best_centers, best_score

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid
    s = 0.14
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.02, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Randomized Grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])
    
    # 7. More structured inits - concentric rings
    np.random.seed(777)
    ring_centers = np.array([[0, 0]])
    for r, n_pts in [(0.1, 6), (0.2, 12), (0.3, 18), (0.4, 24)]:
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        for a in angles:
            ring_centers.append([r*np.cos(a), r*np.sin(a)])
    ring_centers = ring_centers[:N]
    inits.append(ring_centers + np.random.normal(0, 0.01, (N, 2)))
    
    # 8. Fibonacci sphere projected
    np.random.seed(888)
    fib_centers = []
    for i in range(N):
        t = i / N
        phi = np.arccos(1 - 2*t)
        theta = np.pi * (1 + 5**0.5) * i
        x = 0.45 * np.sin(phi) * np.cos(theta)
        y = 0.45 * np.sin(phi) * np.sin(theta)
        fib_centers.append([x, y])
    fib_centers = np.array(fib_centers)
    inits.append(fib_centers)
    
    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 105.0:
            break
            
        centers, score = _optimize_from_start(init, bounds)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
