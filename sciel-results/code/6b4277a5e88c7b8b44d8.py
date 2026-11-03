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
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        # Degenerate facets, skip or handle
        radii = np.full(len(simplices), np.pi)
        centers = np.zeros((len(simplices), 3))
        centers[~valid] = [0, 0, 1]
        return radii, centers, simplices
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Random restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart > 0:
            # Perturb the initial Fibonacci sphere
            np.random.seed(restart * 100)
            points = points + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Annealing parameters
        T = 0.1 # Initial temperature (step size scale)
        cool_rate = 0.995
        no_improve_count = 0
        current_radius = np.pi
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            
            if len(radii) == 0:
                break
                
            max_radius = np.max(radii)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    T *= 0.9
                    if T < 1e-4:
                        break
                if no_improve_count > 200:
                    break
            
            # Identify facets to move
            # Move vertices of the largest facets
            threshold = np.percentile(radii, 80) # Top 20%
            mask = radii > threshold
            
            if not np.any(mask):
                continue
                
            indices_to_move = simplices[mask]
            
            # Accumulate moves for vertices to avoid conflicting updates
            # We want to move each vertex towards the center of the facet it belongs to
            # If a vertex is in multiple "bad" facets, average the target directions?
            # Or just apply sequentially? Sequential is simpler.
            
            # To be more efficient, we can sum up the "pull" vectors for each vertex
            pull = np.zeros((N, 3))
            count = np.zeros(N)
            
            for idx in indices_to_move:
                i_idx, j_idx, k_idx = idx
                # Find the center for this facet
                # We need to map back to the center. 
                # centers array is aligned with simplices.
                # We need to know which center corresponds to which simplex in the masked set.
                # It's easier to just iterate over all facets and check if they are in the mask.
                pass
            
            # Simpler approach: iterate over all facets, if bad, update vertices
            # This might be slow if done naively in Python loop.
            # Let's vectorize where possible or use a list of updates.
            
            updates = np.zeros_like(points)
            
            for m in range(len(simplices)):
                if radii[m] > threshold:
                    idx = simplices[m]
                    i_idx, j_idx, k_idx = idx
                    center = centers[m]
                    
                    # Move towards center
                    # Step size proportional to radius
                    step = T * radii[m]
                    
                    updates[i_idx] += step * (center - points[i_idx])
                    updates[j_idx] += step * (center - points[j_idx])
                    updates[k_idx] += step * (center - points[k_idx])
                    
            points += updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            T *= cool_rate
            
    if best_points is None:
        best_points = points
        
    return best_points
