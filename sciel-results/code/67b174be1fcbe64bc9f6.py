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
    
    # Compute normals (which point to the circumcenter of the spherical triangle)
    # The circumcenter u is parallel to (b-a) x (c-a)
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    # Avoid division by zero
    valid = norms > 1e-10
    if not np.all(valid):
        # If a facet is degenerate, skip it or handle gracefully
        # For now, set norm to 1 and n to [0,0,1] to avoid crash, but this facet is bad
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals to point outwards (away from origin)
    # The circumcenter should be "outside" the triangle relative to the origin?
    # Actually, u * a should be positive (cos theta > 0) for small triangles.
    # If the triangle is large (> 90 deg), cos theta can be negative?
    # Covering radius is usually < 90 deg.
    # To be safe, orient u such that it points towards the triangle.
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius is angle between u and a
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def soft_gradient_step(points, radii, centers, simplices, beta, step_size):
    """
    Performs a gradient step using a soft-max weighting of the facet radii.
    """
    if len(radii) == 0:
        return points
        
    # Calculate weights: w_i = exp(beta * (r_i - r_max))
    r_max = np.max(radii)
    # Shift to prevent underflow/overflow
    exponents = beta * (radii - r_max)
    weights = np.exp(exponents)
    
    # Normalize weights to sum to 1? Or just use as is?
    # Using as is is fine, the step_size will scale it.
    # However, if beta is large, weights are very sparse.
    
    updates = np.zeros_like(points)
    
    # For each facet, add the weighted move to its vertices
    # Move direction for vertex v in facet f is (center_f - v)
    # We want to move v towards center_f.
    
    # Vectorized update:
    # We need to scatter the updates.
    # This is tricky to vectorize perfectly without loops if the structure is irregular,
    # but for N=130 and M~240, a loop is fast enough.
    
    for i in range(len(simplices)):
        w = weights[i]
        if w < 1e-10:
            continue
        simplex = simplices[i]
        center = centers[i]
        
        # Move each vertex towards the center
        # The force is proportional to w * (center - vertex)
        # We accumulate this into the update vector for that vertex
        
        for v_idx in simplex:
            updates[v_idx] += w * (center - points[v_idx])
    
    points += step_size * updates
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    def evaluate(points):
        radii, _, _ = compute_facet_data(points)
        if len(radii) == 0:
            return np.pi
        return np.max(radii)
        
    # Initializations
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(init_fib)
    
    # 2. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    inits.append(init_rand)
    
    # 3. Perturbed Fibonacci
    for noise_scale in [0.1, 0.2, 0.3]:
        for seed in [1, 2, 3]:
            np.random.seed(seed)
            p = init_fib + np.random.randn(N, 3) * noise_scale
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            inits.append(p)
            
    # Optimization parameters
    beta_start = 10.0
    beta_end = 50.0
    step_start = 0.05
    step_end = 0.005
    
    for restart_idx, init_points in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        points = init_points.copy()
        
        # Schedule for beta and step size
        # We can use a fixed beta or anneal it.
        # Let's try a few runs with different beta strategies
        
        for beta_val in [5.0, 10.0, 20.0, 50.0]:
            if time.time() - start_time > time_limit:
                break
                
            current_points = points.copy()
            current_radius = evaluate(current_points)
            best_local_radius = current_radius
            best_local_points = current_points.copy()
            
            no_improve = 0
            step_size = 0.05
            
            for step in range(2000):
                if time.time() - start_time > time_limit:
                    break
                    
                radii, centers, simplices = compute_facet_data(current_points)
                if len(radii) == 0:
                    break
                    
                current_radius = np.max(radii)
                
                if current_radius < best_local_radius:
                    best_local_radius = current_radius
                    best_local_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 100:
                        # Decay step size
                        step_size *= 0.9
                        if step_size < 1e-4:
                            break
                    if no_improve > 300:
                        break
                        
                current_points = soft_gradient_step(current_points, radii, centers, simplices, beta_val, step_size)
                
            # Update global best
            if best_local_radius < best_radius:
                best_radius = best_local_radius
                best_points = best_local_points.copy()
                
            # If we found a good solution, restart from there?
            # Or just continue to next init.
            # Let's use the best local point as the new starting point for the next beta?
            # No, let's just use the best found so far for the next init if it's better.
            if best_local_radius < evaluate(points):
                points = best_local_points.copy()

    if best_points is None:
        best_points = inits[0]
        
    return best_points
