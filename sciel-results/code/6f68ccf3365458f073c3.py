import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_facet_data(points):
    """
    Computes the circumradius and circumcenter direction for each facet of the convex hull.
    Returns:
        radii: array of shape (M,)
        centers: array of shape (M, 3) unit vectors
        simplices: array of shape (M, 3)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    # Extract vertices for all facets
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    # Avoid division by zero
    valid = norms > 1e-10
    if not np.all(valid):
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals to point outwards (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius is angle between u and a
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def lloyd_relax(points, n_iter=50):
    """
    Lloyd's algorithm (Voronoi relaxation) for sphere covering.
    Assigns each point to its Voronoi cell, then moves it to the centroid of the cell.
    This globally redistributes points to equalize cell areas/volumes.
    """
    points = points.copy()
    for it in range(n_iter):
        # Compute facet data
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        # For each point, find the "center of mass" of its Voronoi cell
        # Approximation: for each point, average the circumcenters of all facets that contain it
        cell_centers = np.zeros_like(points)
        cell_counts = np.zeros(N)
        
        for m in range(len(simplices)):
            simplex = simplices[m]
            center = centers[m]
            # Weight by facet area (proportional to circumradius squared or just 1)
            weight = 1.0
            for idx in simplex:
                cell_centers[idx] += weight * center
                cell_counts[idx] += weight
        
        # Normalize
        valid_counts = cell_counts > 1e-10
        for i in range(N):
            if valid_counts[i]:
                new_point = cell_centers[i] / np.linalg.norm(cell_centers[i])
                # Blend with original point to avoid large jumps
                points[i] = 0.5 * points[i] + 0.5 * new_point
                points[i] /= np.linalg.norm(points[i])
        
    return points

def greedy_init(n_points, seed=0):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Generate many candidates
        candidates = np.random.randn(1000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute min distance to existing points
        existing = np.array(points)
        # Dot products
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
    
    return np.array(points)

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random points
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    
    # 3. Greedy initialization
    init_greedy = greedy_init(N, seed=123)
    
    # 4. Fibonacci + small noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.1
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    
    initializations = [init_fib, init_rand, init_greedy, init_fib_noise]
    
    # Add more randomized restarts
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Random perturbation of best found so far, or Fibonacci
            if best_points is not None and restart % 3 == 0:
                base = best_points
            else:
                base = init_fib
            np.random.seed(restart * 100 + 7)
            points = base + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Phase 1: Lloyd's relaxation for global redistribution
        if time.time() - start_time < time_limit - 5.0:
            points = lloyd_relax(points, n_iter=30)
        
        # Check radius after Lloyd
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) > 0:
            current_radius = np.max(radii)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
        # Phase 2: Local facet-pushing optimization
        alpha = 0.05
        no_improve = 0
        current_radius = np.pi
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            max_radius, radii, simplices, centers = compute_facet_data(points)
            
            if len(radii) == 0:
                break
                
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 50:
                    alpha *= 0.9
                    if alpha < 1e-4:
                        break
                if no_improve > 200:
                    break
            
            # Identify facets to move - use top 20%
            threshold = np.percentile(radii, 80)
            mask = radii > threshold
            
            if not np.any(mask):
                # If no facets are above threshold, just pick the largest
                max_idx = np.argmax(radii)
                mask = np.zeros(len(radii), dtype=bool)
                mask[max_idx] = True
            
            # Accumulate moves for vertices to avoid conflicting updates
            updates = np.zeros_like(points)
            
            for m in np.where(mask)[0]:
                simplex = simplices[m]
                i_idx, j_idx, k_idx = simplex
                
                a = points[i_idx]
                b = points[j_idx]
                c = points[k_idx]
                
                center = centers[m]
                
                # Move towards center
                # Step size proportional to radius
                step_size = alpha * radii[m]
                
                updates[i_idx] += step_size * (center - a)
                updates[j_idx] += step_size * (center - b)
                updates[k_idx] += step_size * (center - c)
            
            points += updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Occasionally run another Lloyd pass if time permits
        if time.time() - start_time < time_limit - 10.0 and restart % 5 == 0:
            points = lloyd_relax(points, n_iter=20)
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) > 0:
                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
    
    if best_points is None:
        best_points = init_fib
        
    return best_points
