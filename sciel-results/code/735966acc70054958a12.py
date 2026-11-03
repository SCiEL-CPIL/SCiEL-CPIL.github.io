"""Improved program using Delaunay-based sparse optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
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
        # Fallback to all pairs if Delaunay fails (e.g., collinear points)
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _optimize_from_start(start_centers, bounds, time_limit=15.0):
    """
    Optimization using Delaunay-aware sparse objective.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    # Helper to compute sparse objective
    def objective(params, k):
        centers = params.reshape(N, 2)
        # Keep centers within bounds to avoid weird gradients
        centers = np.clip(centers, -0.5, 0.5)
        
        # Compute Delaunay edges
        edges = _get_delaunay_edges(centers)
        
        # Collect distances for active constraints
        active_dists = []
        
        # Pairwise distances for Delaunay edges
        for i, j in edges:
            d = np.linalg.norm(centers[i] - centers[j])
            active_dists.append(d / 2.0)
            
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        active_dists.extend(wall_dists.flatten())
        
        active_dists = np.array(active_dists)
        
        if len(active_dists) == 0:
            return 0.0
            
        m = np.min(active_dists)
        # Soft min to make it differentiable
        shifted = active_dists - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min

    # Use L-BFGS-B with the sparse objective
    start_time = time.time()
    
    # Stage 1: Coarse
    try:
        result = minimize(
            objective, 
            best_centers.reshape(-1), 
            args=(100.0,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        centers = result.x.reshape(N, 2)
        centers = np.clip(centers, -0.5, 0.5)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
        
    # Stage 2: Fine
    if time.time() - start_time < time_limit:
        try:
            result = minimize(
                objective, 
                best_centers.reshape(-1), 
                args=(500.0,),
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
        except Exception:
            pass
            
    # Stage 3: Ultra Fine
    if time.time() - start_time < time_limit:
        try:
            result = minimize(
                objective, 
                best_centers.reshape(-1), 
                args=(2000.0,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False}
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

def _hexagonal_init(n_points, seed=42):
    """Generate a hexagonal grid initialization."""
    np.random.seed(seed)
    # Approximate spacing for 50 points in unit square
    # Area per point ~ 1/50 = 0.02. Hex area ~ s^2 * sqrt(3)/2.
    # s^2 ~ 0.02 * 2 / 1.732 ~ 0.023. s ~ 0.15.
    s = 0.15
    points = []
    # Generate a large grid
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    
    # Filter points inside the square
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    
    if len(points) < n_points:
        # Fallback to random if grid is too sparse
        return np.random.uniform(-0.45, 0.45, (n_points, 2))
        
    # Select n_points closest to the center to maximize wall distance
    dist_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dist_to_center)[:n_points]
    return points[indices]

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid (centered)
    inits.append(_hexagonal_init(N, seed=42))
    
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
    hex_init = _hexagonal_init(N, seed=42)
    jittered = hex_init + np.random.normal(0, 0.02, (N, 2))
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
        
    # 7. Hex with different seed
    inits.append(_hexagonal_init(N, seed=1234))

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 110.0: 
            break
            
        # Allocate remaining time / remaining inits
        remaining_inits = len(inits) - i
        time_per_init = min(15.0, (115.0 - (time.time() - start_time)) / remaining_inits)
        
        centers, score = _optimize_from_start(init, bounds, time_limit=time_per_init)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    if best_centers is None:
        # Fallback
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
