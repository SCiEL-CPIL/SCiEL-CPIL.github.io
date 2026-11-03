import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _covering_radius(points):
    """Compute the covering radius in radians."""
    try:
        hull = ConvexHull(points)
        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Compute circumcenter
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
                
            rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            
            if rad > max_rad:
                max_rad = rad
                
    except Exception:
        return np.pi # Fallback, large radius
        
    return max_rad

def construct():
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    
    start_time = time.time()
    max_time = 110.0 # Safety margin
    
    step_size = 0.05
    min_step_size = 0.001
    fail_count = 0
    
    while time.time() - start_time < max_time:
        prev_points = points.copy()
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        # Find all facet radii and indices
        facet_radii = []
        facet_indices = []
        for idx, simplex in enumerate(hull.simplices):
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
            rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            facet_radii.append(rad)
            facet_indices.append(simplex)
            
        if not facet_radii:
            break
            
        # Sort facets by radius (descending)
        sorted_indices = np.argsort(facet_radii)[::-1]
        
        # Move vertices of the top K worst facets
        K = 5
        moved_indices = set()
        for k in range(min(K, len(sorted_indices))):
            facet_idx = sorted_indices[k]
            simplex = facet_indices[facet_idx]
            
            # Recompute circumcenter for this specific facet to be sure
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
                
            # Move vertices towards circumcenter
            for j in range(3):
                v_idx = simplex[j]
                if v_idx in moved_indices:
                    continue
                v = points[v_idx]
                new_v = v + step_size * (c - v)
                norm_v = np.linalg.norm(new_v)
                if norm_v > 1e-10:
                    points[v_idx] = new_v / norm_v
                    moved_indices.add(v_idx)
                    
        # Calculate new covering radius
        current_rad = _covering_radius(points)
        
        if current_rad < best_rad:
            best_rad = current_rad
            best_points = points.copy()
            fail_count = 0
            # Optionally increase step size slightly
            # step_size = min(step_size * 1.05, 0.1)
        else:
            points = prev_points
            fail_count += 1
            if fail_count > 5:
                step_size *= 0.8
                fail_count = 0
                if step_size < min_step_size:
                    break

    # Final Polish
    if time.time() - start_time < max_time - 10:
        points = best_points.copy()
        step_size = 0.01
        for _ in range(50):
            if time.time() - start_time > max_time - 5:
                break
            prev_points = points.copy()
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            max_facet_rad = 0.0
            worst_facet_idx = -1
            worst_circumcenter = None
            worst_indices = None
            for idx, simplex in enumerate(hull.simplices):
                v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                n = np.cross(v1 - v2, v1 - v3)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
                if rad > max_facet_rad:
                    max_facet_rad = rad
                    worst_facet_idx = idx
                    worst_circumcenter = c
                    worst_indices = simplex
            if worst_facet_idx == -1:
                break
            for j in range(3):
                v_idx = worst_indices[j]
                v = points[v_idx]
                new_v = v + step_size * (worst_circumcenter - v)
                points[v_idx] = new_v / np.linalg.norm(new_v)
            current_rad = _covering_radius(points)
            if current_rad < best_rad:
                best_rad = current_rad
                best_points = points.copy()
            else:
                points = prev_points

    return best_points
