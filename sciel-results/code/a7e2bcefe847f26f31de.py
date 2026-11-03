import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi, np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.pi, np.array([]), np.array([])
        
    # Extract vertices for all facets
    # simplices is shape (M, 3)
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals: cross(b-a, c-a)
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    # Avoid division by zero
    valid = norms > 1e-10
    if not np.all(valid):
        # Handle degenerate cases (though unlikely for convex hull)
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals to point outwards (away from origin)
    # Centroid of each triangle
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius is angle between u and a
    # cos_theta = dot(u, a)
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    max_radius = np.max(radii)
    return max_radius, radii, simplices

def soft_max_objective(points, temperature):
    """
    Compute a differentiable approximation of the max covering radius using soft-max.
    Returns the objective value and gradient with respect to points.
    """
    # Pairwise dot products: (N, N)
    dots = points @ points.T
    # Angular distances: arccos of dot products
    dots_clipped = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots_clipped)
    
    # For each point i, find the min angle to any other point
    # But we want the max over the sphere, which is approximated by 
    # considering the Voronoi cells. A simpler approach: 
    # Use the fact that the covering radius is related to the minimum
    # distance from any point on the sphere to the nearest configuration point.
    
    # Alternative: Use a grid of test points on the sphere
    # This is expensive but differentiable if we use smooth approximations
    
    # Better approach: Use the convex hull facets but make it differentiable
    # by using a smooth max over the facet circumradii
    
    # For now, let's use a simpler potential: minimize the sum of exp(beta * d_i)
    # where d_i is the distance from point i to its nearest neighbor
    # This encourages spreading out
    
    # Compute nearest neighbor distances
    np.fill_diagonal(angles, np.pi)  # Exclude self
    min_angles = np.min(angles, axis=1)
    
    # Soft max of min_angles
    beta = 1.0 / temperature
    exp_beta_min = np.exp(beta * min_angles)
    soft_max = (1.0 / beta) * np.log(np.sum(exp_beta_min))
    
    # Gradient computation
    # This is complex, so let's use numerical gradients or a simpler approach
    
    return soft_max

def construct():
    start_time = time.time()
    time_limit = 115.0
    
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
    
    # 3. Fibonacci + small noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.1
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    
    initializations = [init_fib, init_rand, init_fib_noise]
    
    # Add more randomized restarts based on time
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Random perturbation of Fibonacci
            np.random.seed(restart * 100 + 7)
            noise_scale = 0.2 + 0.1 * np.sin(restart)
            points = init_fib + np.random.randn(N, 3) * noise_scale
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Gradient-based optimization with dynamic temperature
        temperature = 0.1
        lr = 0.01
        no_improve = 0
        current_best_for_restart = np.pi
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
            
            # Compute actual covering radius for tracking
            max_radius, radii, simplices = compute_covering_radius_and_facets(points)
            
            if len(simplices) == 0:
                break
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve = 0
            elif max_radius < current_best_for_restart:
                current_best_for_restart = max_radius
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    temperature *= 0.95
                    lr *= 0.95
                    if lr < 1e-5:
                        break
                if no_improve > 100:
                    break
            
            # Compute gradient using the soft-max objective
            # For efficiency, we'll use a simplified gradient approach:
            # Move points away from each other based on their local density
            
            # Compute pairwise distances
            dots = points @ points.T
            dots_clipped = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots_clipped)
            
            # For each point, compute the force from neighbors
            # Points should move away from their nearest neighbors
            np.fill_diagonal(angles, np.pi)
            
            # Weight by inverse distance (closer points have stronger repulsion)
            weights = np.exp(-angles / temperature)
            np.fill_diagonal(weights, 0)
            
            # Force on each point: sum of (p_i - p_j) * w_ij
            # This is like a repulsive force
            forces = np.zeros_like(points)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        forces[i] += weights[i, j] * (points[i] - points[j])
            
            # Normalize forces
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            forces = forces / force_norms
            
            # Update points
            points += lr * forces
            points /= np.linalg.norm(points, axis=1, keepdims=True)

    if best_points is None:
        best_points = init_fib
        
    return best_points
