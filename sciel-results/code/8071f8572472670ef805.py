import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization loop
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for safety
    
    # Parameters
    eta = 0.02  # Smaller step size for stability
    threshold_factor = 0.98  # Consider facets with radius > 98% of max
    jitter_interval = 50
    iteration = 0
    
    best_points = points.copy()
    best_radius = np.inf

    while time.time() < deadline:
        iteration += 1
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, try a small random perturbation
            points += np.random.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        simplices = hull.simplices
        num_faces = len(simplices)
        
        radii = np.zeros(num_faces)
        centers = np.zeros((num_faces, 3))
        
        for k in range(num_faces):
            idx = simplices[k]
            a = points[idx[0]]
            b = points[idx[1]]
            c = points[idx[2]]
            
            v1 = b - a
            v2 = c - a
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            
            if n_norm < 1e-10:
                radii[k] = np.inf
                continue
                
            c0 = n / n_norm
            centroid = (a + b + c) / 3.0
            
            if np.dot(c0, centroid) < 0:
                c0 = -c0
                
            centers[k] = c0
            # Chord distance
            radii[k] = np.linalg.norm(c0 - a)
            
        max_radius = np.max(radii)
        
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
            
        # Find worst faces
        threshold = max_radius * threshold_factor
        worst_indices = np.where(radii > threshold)[0]
        
        if len(worst_indices) == 0:
            break
            
        # Update points
        updates = np.zeros_like(points)
        counts = np.zeros(N)
        
        for k in worst_indices:
            idx = simplices[k]
            c0 = centers[k]
            for i in idx:
                updates[i] += (c0 - points[i])
                counts[i] += 1
                
        mask = counts > 0
        points[mask] += eta * updates[mask] / counts[mask, None]
        
        # Normalize
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Add jitter periodically to escape local minima
        if iteration % jitter_interval == 0:
            points += np.random.normal(0, 0.005, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
    return best_points
