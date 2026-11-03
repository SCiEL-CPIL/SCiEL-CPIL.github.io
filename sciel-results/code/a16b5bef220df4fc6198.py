import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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
    
    max_radius = np.max(radii)
    return max_radius, radii, simplices, u

def polish_largest_facets(points, max_facets=20, max_iter=500):
    """
    Polish the configuration by minimizing the max facet radius using Nelder-Mead
    on the vertices of the largest facets.
    """
    best_points = points.copy()
    best_radius, radii, simplices, centers = compute_covering_radius_and_facets(points)
    
    if len(simplices) == 0:
        return best_points
    
    # Get indices of vertices in the largest facets
    top_k = min(max_facets, len(radii))
    top_indices = np.argsort(radii)[-top_k:]
    
    # Collect unique vertex indices from these facets
    vertex_indices = set()
    for idx in top_indices:
        for v in simplices[idx]:
            vertex_indices.add(v)
    
    vertex_indices = list(vertex_indices)
    
    if len(vertex_indices) == 0:
        return best_points
    
    # Objective: maximize the minimum angular distance (minimize max radius)
    # We'll minimize the max radius over all facets
    def objective(x_flat):
        x = x_flat.reshape(-1, 3)
        # Reconstruct full point array
        new_points = points.copy()
        for i, idx in enumerate(vertex_indices):
            new_points[idx] = x[i]
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points /= norms
        # Check for degeneracy
        if not np.all(np.isfinite(new_points)):
            return np.pi
        r, _, _, _ = compute_covering_radius_and_facets(new_points)
        return r
    
    x0 = points[vertex_indices].flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_radius:
            x_opt = result.x.reshape(-1, 3)
            best_points = points.copy()
            for i, idx in enumerate(vertex_indices):
                best_points[idx] = x_opt[i]
            best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
            best_radius = result.fun
    except Exception:
        pass
    
    return best_points

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
    
    # 4. Tammes-like initial: start with icosahedron vertices and subdivide
    # Use a better structured start
    np.random.seed(7)
    init_struct = init_fib + np.random.randn(N, 3) * 0.05
    init_struct /= np.linalg.norm(init_struct, axis=1, keepdims=True)
    
    initializations = [init_fib, init_rand, init_fib_noise, init_struct]
    
    # Add more randomized restarts based on time
    max_restarts = 15
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 20.0:  # Leave time for polish
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Random perturbation of Fibonacci
            np.random.seed(restart * 100 + 7)
            noise_scale = 0.2 + 0.1 * (restart % 3)
            points = init_fib + np.random.randn(N, 3) * noise_scale
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Local Optimization with adaptive step size
        alpha = 0.1
        no_improve = 0
        best_local_radius = np.pi
        
        for step in range(5000):
            if time.time() - start_time > time_limit - 20.0:
                break
                
            max_radius, radii, simplices, centers = compute_covering_radius_and_facets(points)
            
            if len(simplices) == 0:
                break
                
            if max_radius < best_local_radius:
                best_local_radius = max_radius
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    alpha *= 0.85
                    if alpha < 1e-5:
                        break
                if no_improve > 150:
                    break
            
            # Identify facets to move - top 15%
            threshold = np.percentile(radii, 85)
            mask = radii > threshold
            
            if not np.any(mask):
                max_idx = np.argmax(radii)
                mask = np.zeros(len(radii), dtype=bool)
                mask[max_idx] = True
            
            # Accumulate moves for vertices
            updates = np.zeros_like(points)
            
            for m in np.where(mask)[0]:
                simplex = simplices[m]
                i_idx, j_idx, k_idx = simplex
                
                a = points[i_idx]
                b = points[j_idx]
                c = points[k_idx]
                
                center = centers[m]
                
                # Step size proportional to radius, with damping
                step_size = alpha * min(radii[m], 0.3)
                
                updates[i_idx] += step_size * (center - a)
                updates[j_idx] += step_size * (center - b)
                updates[k_idx] += step_size * (center - c)
            
            points += updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Polish the best local configuration
        if best_local_radius < best_radius:
            polished = polish_largest_facets(points, max_facets=25, max_iter=300)
            pol_radius, _, _, _ = compute_covering_radius_and_facets(polished)
            if pol_radius < best_radius:
                best_radius = pol_radius
                best_points = polished.copy()
            elif best_local_radius < best_radius:
                best_radius = best_local_radius
                best_points = points.copy()
        else:
            # Still try to polish even if not better than global best
            polished = polish_largest_facets(points, max_facets=20, max_iter=200)
            pol_radius, _, _, _ = compute_covering_radius_and_facets(polished)
            if pol_radius < best_radius:
                best_radius = pol_radius
                best_points = polished.copy()

    # Final polish on the best configuration found
    if best_points is not None and time.time() - start_time < time_limit - 5.0:
        final_polished = polish_largest_facets(best_points, max_facets=30, max_iter=500)
        final_radius, _, _, _ = compute_covering_radius_and_facets(final_polished)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_polished.copy()

    if best_points is None:
        best_points = init_fib
        
    return best_points
