import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_score = np.inf
    
    beta = 10.0
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Map points to facets
        point_to_facets = [[] for _ in range(N)]
        for idx, simplex in enumerate(hull.simplices):
            for v in simplex:
                point_to_facets[v].append(idx)
                
        new_pts = np.zeros_like(pts)
        current_max_rad = 0.0
        
        for i in range(N):
            facet_indices = point_to_facets[i]
            if not facet_indices:
                new_pts[i] = pts[i]
                continue
                
            centers = []
            max_d2 = 0.0
            for f_idx in facet_indices:
                simplex = hull.simplices[f_idx]
                a = pts[simplex[0]]
                b = pts[simplex[1]]
                c = pts[simplex[2]]
                u = a - b
                v = b - c
                w = np.cross(u, v)
                norm = np.linalg.norm(w)
                if norm > 1e-10:
                    c_center = w / norm
                    centers.append(c_center)
                    # Distance squared from pts[i] to c_center
                    dot = np.dot(centers[-1], pts[i])
                    d2 = 2.0 - 2.0 * dot
                    if d2 > max_d2:
                        max_d2 = d2
                        
            if not centers:
                new_pts[i] = pts[i]
                continue
                
            centers = np.array(centers)
            current_max_rad = max(current_max_rad, np.sqrt(max_d2))
            
            dots = np.dot(centers, pts[i])
            d2s = 2.0 - 2.0 * dots
            
            weights = np.exp(beta * d2s)
            
            centroid = np.sum(weights[:, np.newaxis] * centers, axis=0)
            norm_c = np.linalg.norm(centroid)
            if norm_c > 1e-10:
                new_pts[i] = centroid / norm_c
            else:
                new_pts[i] = pts[i]
                
        pts = new_pts
        
        # Update best
        # Note: current_max_rad is an approximation (max over vertices of Voronoi cells)
        # It is a lower bound on the true covering radius?
        # Actually, the true covering radius is the max over the whole sphere.
        # The vertices of the Voronoi cells are the points farthest from the site *within that cell*.
        # So the max over all vertices is exactly the covering radius?
        # Yes, because every point on the sphere is in some Voronoi cell, and the farthest point in that cell from the site is a vertex.
        # So current_max_rad IS the covering radius.
        
        if current_max_rad < best_score:
            best_score = current_max_rad
            best_pts = pts.copy()
            
    return best_pts
