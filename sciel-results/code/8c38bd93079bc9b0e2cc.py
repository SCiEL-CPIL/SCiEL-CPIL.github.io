import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.pi # Max possible radius
    
    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    alpha = 0.05 # Step size
    decay = 0.995
    
    iter_count = 0
    
    while time.time() - start_time < time_limit:
        # Compute Convex Hull
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            # If hull fails, try a small random perturbation
            points += np.random.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        # Calculate circumradius for each facet
        radii = np.zeros(len(facets))
        centers = np.zeros((len(facets), 3))
        
        for idx, f in enumerate(facets):
            a = points[f[0]]
            b = points[f[1]]
            c = points[f[2]]
            
            # Normal to the plane
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                radii[idx] = 0 # Degenerate
                continue
                
            m = n / norm_n
            
            # Ensure m is on the same side as a (cos r > 0)
            # This picks the circumcenter that is "close" to the facet
            if np.dot(m, a) < 0:
                m = -m
                
            radii[idx] = np.arccos(np.clip(np.dot(m, a), -1.0, 1.0))
            centers[idx] = m
            
        current_radius = np.max(radii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Identify worst facets
        # Use a threshold to select facets contributing to the max radius
        # Select facets with radius > current_radius - 0.05
        threshold = current_radius - 0.05
        worst_indices = np.where(radii > threshold)[0]
        
        if len(worst_indices) == 0:
            worst_indices = np.argmax(radii)
            worst_indices = np.array([worst_indices])
            
        # Update points
        updates = np.zeros_like(points)
        
        for idx in worst_indices:
            c = centers[idx]
            f = facets[idx]
            for k in f:
                updates[k] += c - points[k]
                
        # Apply updates
        if np.any(updates != 0):
            points += alpha * updates
            # Normalize
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points /= norms
            
        # Decay step size
        alpha *= decay
        iter_count += 1

    # Final normalization
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
