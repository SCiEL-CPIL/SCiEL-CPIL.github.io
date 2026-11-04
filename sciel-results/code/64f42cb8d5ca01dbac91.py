import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_radius = np.inf
    best_points = points.copy()
    
    # Parameters for local search
    step_size = 0.05
    min_step_size = 0.001
    decay = 0.999
    
    # Optimization loop
    iter_count = 0
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Compute Convex Hull
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails (e.g., points not in general position or degenerate), break or perturb
            # Perturb points slightly
            points += np.random.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        facets = hull.simplices
        
        max_radius = 0
        max_facet_idx = -1
        max_center = None
        max_vertices = None
        
        # Evaluate circumradius for each facet
        # Facets are triangles (i, j, k)
        for idx, f in enumerate(facets):
            v0 = points[f[0]]
            v1 = points[f[1]]
            v2 = points[f[2]]
            
            # Compute circumcenter
            # m is parallel to (v0 - v1) x (v1 - v2)
            u = np.cross(v0 - v1, v1 - v2)
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-10:
                continue
            m = u / norm_u
            
            # Ensure m is on the same side as the vertices (dot product > 0)
            # The circumcenter of a small spherical triangle is close to the vertices.
            if np.dot(m, v0) < 0:
                m = -m
            
            # Circumradius is arccos(m . v0)
            cos_r = np.clip(np.dot(m, v0), -1.0, 1.0)
            radius = np.arccos(cos_r)
            
            if radius > max_radius:
                max_radius = radius
                max_facet_idx = idx
                max_center = m
                max_vertices = [f[0], f[1], f[2]]
                
        if max_facet_idx == -1:
            break
            
        # Update vertices of the worst facet
        # Move them towards the circumcenter
        # p_new = normalize(p + alpha * (m - p))
        # Note: m is a unit vector. p is a unit vector.
        # The vector (m - p) points from p towards m.
        
        for v_idx in max_vertices:
            p = points[v_idx]
            # Move towards center
            new_p = p + step_size * (max_center - p)
            norm_new_p = np.linalg.norm(new_p)
            if norm_new_p > 1e-10:
                points[v_idx] = new_p / norm_new_p
            else:
                # If it collapsed to origin, push it out randomly? 
                # Unlikely with small step size, but handle it.
                points[v_idx] = p + np.random.normal(0, 0.1, 3)
                points[v_idx] /= np.linalg.norm(points[v_idx])
        
        # Decay step size
        step_size *= decay
        if step_size < min_step_size:
            step_size = min_step_size
            
        # Check time more frequently in inner loop if needed, but outer loop check is sufficient
        # If we are improving, keep going.
        # We track best points based on max_radius
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
            
        # If step size is very small and improvement is negligible, we might be stuck.
        # But the decay handles this.
        
    # Final cleanup: ensure points are on sphere
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
