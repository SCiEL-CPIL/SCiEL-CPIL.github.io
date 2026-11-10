import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _spherical_mean(points):
    """Calculate the spherical mean of a set of points."""
    if len(points) == 0:
        return np.zeros(3)
    return points.mean(axis=0)

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros(3)
    return v / norm

def _spherical_kmeans(points, data_points, max_iters=100, tol=1e-6):
    """
    Spherical K-Means to find N centroids that minimize mean distance to data_points.
    data_points: (M, 3) array of points on the sphere.
    points: (N, 3) initial centroids.
    """
    M = len(data_points)
    N = len(points)
    
    # Precompute data points
    # Assignment step
    # dist[i, j] = 1 - dot(data[i], centroid[j])
    
    centroids = points.copy()
    
    for _ in range(max_iters):
        # Assignment
        # dist = 1 - data_points @ centroids.T
        dists = 1.0 - np.dot(data_points, centroids.T)
        labels = np.argmin(dists, axis=1)
        
        # Update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(N)
        
        for i in range(N):
            mask = labels == i
            if np.any(mask):
                new_centroids[i] = np.sum(data_points[mask], axis=0)
                counts[i] = np.sum(mask)
            else:
                # Reinitialize if empty
                new_centroids[i] = centroids[i] + np.random.normal(0, 0.1, 3)
                counts[i] = 1
        
        new_centroids = np.array([_normalize(c) for c in new_centroids])
        
        # Check convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
            
        centroids = new_centroids
        
    return centroids

def _calculate_covering_radius(points):
    """
    Calculate the covering radius of the configuration.
    Returns the radius in radians and the worst face info.
    """
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
    except Exception:
        return np.pi, None, None
        
    max_radius = 0.0
    max_face_idx = -1
    max_circumcenter = None
    
    for idx, face in enumerate(faces):
        v1, v2, v3 = points[face]
        
        # Calculate circumcenter
        u = v2 - v1
        w = v3 - v1
        n = np.cross(u, w)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        # Planar circumcenter
        A = np.array([[np.dot(u, u), np.dot(w, u)], [np.dot(u, w), np.dot(w, w)]])
        B = np.array([np.dot(u, u), np.dot(w, w)])
        
        try:
            coeffs = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            continue
        a, b = coeffs
        c_p = v1 + a * u + b * w
        
        c_p_norm = np.linalg.norm(c_p)
        if c_p_norm < 1e-10:
            continue
            
        c_s = c_p / c_p_norm
        cos_R = np.dot(c_s, v1)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        if R > max_radius:
            max_radius = R
            max_face_idx = idx
            max_circumcenter = c_s
            
    return max_radius, max_face_idx, max_circumcenter

def _local_search(points, max_time=5.0):
    """
    Local search to minimize covering radius.
    """
    start_time = time.time()
    step_size = 0.1
    decay = 0.98
    
    best_points = points.copy()
    best_radius = _calculate_covering_radius(points)[0]
    
    while time.time() - start_time < max_time:
        radius, face_idx, circumcenter = _calculate_covering_radius(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
        if face_idx == -1:
            break
            
        faces = ConvexHull(points).simplices
        face = faces[face_idx]
        
        # Move vertices of the worst face towards the circumcenter
        for i in face:
            v = points[i]
            # Move towards circumcenter on the sphere
            # Simple linear move then normalize
            points[i] = v + step_size * (circumcenter - v)
            
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        step_size *= decay
        
    return best_points

def construct():
    start_time = time.time()
    max_time = 115.0
    
    # Generate a dense set of points on the sphere for K-Means
    # Use Fibonacci spiral for even distribution
    M = 2000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    data_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_global_points = None
    best_global_radius = np.pi
    
    np.random.seed(42) # For reproducibility of the restarts
    
    while time.time() - start_time < max_time:
        # Random initialization
        random_points = np.random.randn(N, 3)
        random_points = np.array([_normalize(p) for p in random_points])
        
        # K-Means phase
        km_points = _spherical_kmeans(random_points, data_points, max_iters=50)
        
        # Local search phase
        # Allocate remaining time budget
        elapsed = time.time() - start_time
        remaining = max_time - elapsed
        if remaining < 10:
            break
            
        # Spend a portion of remaining time on local search
        # We want to do a few restarts, so limit local search time
        local_time_budget = min(10.0, remaining / 3.0)
        
        refined_points = _local_search(km_points, max_time=local_time_budget)
        
        # Evaluate
        radius, _, _ = _calculate_covering_radius(refined_points)
        
        if radius < best_global_radius:
            best_global_radius = radius
            best_global_points = refined_points.copy()
            
        # If we found a very good solution, we can break early?
        # The record is 18.3 degrees = 0.3194 radians.
        # If we get below 0.32, it's excellent.
        if best_global_radius < 0.32:
            break

    if best_global_points is None:
        # Fallback to Fibonacci if something failed
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_global_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_global_points
