import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            dot_val_neg = np.clip(np.dot(-c, p1), -1.0, 1.0)
            dist_neg = np.arccos(dot_val_neg)
            
            if dist_neg < dist:
                dist = dist_neg
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return np.pi / 2
        
    return np.degrees(max_radius)

def _get_worst_facet_info(points):
    """
    Find the facet with the largest circumradius.
    Returns (worst_radius, worst_vertices, worst_circumcenter) or (0.0, None, None).
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        worst_radius = 0.0
        worst_vertices = None
        worst_circumcenter = None
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            dot_val_neg = np.clip(np.dot(-c, p1), -1.0, 1.0)
            dist_neg = np.arccos(dot_val_neg)
            
            if dist_neg < dist:
                dist = dist_neg
                c = -c
            
            if dist > worst_radius:
                worst_radius = dist
                worst_vertices = tri
                worst_circumcenter = c
                
        return worst_radius, worst_vertices, worst_circumcenter
    except Exception:
        return 0.0, None, None

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    step_size = 0.05
    min_step = 1e-5
    max_iter = 10000
    
    iter_count = 0
    consecutive_failures = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        worst_radius, worst_vertices, worst_circumcenter = _get_worst_facet_info(points)
        
        if worst_vertices is None:
            break
            
        # Save old positions
        old_positions = points[worst_vertices].copy()
        
        # Move vertices towards the circumcenter
        for v_idx in worst_vertices:
            p = points[v_idx]
            direction = worst_circumcenter - p
            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-12:
                continue
            direction /= norm_dir
            
            points[v_idx] = points[v_idx] + step_size * direction
            points[v_idx] /= np.linalg.norm(points[v_idx])
        
        # Check improvement
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            consecutive_failures = 0
        else:
            # Revert
            points[worst_vertices] = old_positions
            consecutive_failures += 1
            step_size *= 0.9
            
            if step_size < min_step:
                # Try random kick
                if consecutive_failures > 50:
                    idx = np.random.randint(N)
                    old_p = points[idx].copy()
                    rand_dir = np.random.randn(3)
                    rand_dir /= np.linalg.norm(rand_dir)
                    points[idx] = points[idx] + 0.01 * rand_dir
                    points[idx] /= np.linalg.norm(points[idx])
                    
                    new_rad = _compute_covering_radius(points)
                    if new_rad < best_radius:
                        best_radius = new_rad
                        best_points = points.copy()
                        consecutive_failures = 0
                        step_size = 0.01
                    else:
                        points[idx] = old_p
                        break
                else:
                    break

    return best_points
