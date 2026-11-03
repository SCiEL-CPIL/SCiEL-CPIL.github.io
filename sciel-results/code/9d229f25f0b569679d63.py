import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Helper to normalize points
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    # Helper to compute covering radius and worst facet
    def get_covering_radius_and_worst(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            worst_simplex = None
            worst_center = None
            facet_radii = np.zeros(len(hull.simplices))
            
            for i, simplex in enumerate(hull.simplices):
                facet_pts = pts[simplex]
                v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
                
                n = np.cross(v1 - v0, v2 - v0)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n /= n_norm
                
                if n @ v0 < 0:
                    n = -n
                
                dist = np.arccos(np.clip(n @ v0, -1, 1))
                facet_radii[i] = dist
                
                if dist > max_rad:
                    max_rad = dist
                    worst_simplex = simplex
                    worst_center = n
                    
            return max_rad, worst_simplex, worst_center, facet_radii
        except Exception:
            return np.pi, None, None, None

    # Initial points: Fibonacci Spiral
    def init_fibonacci():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return normalize(points)

    best_points = init_fibonacci()
    best_radius, _, _, _ = get_covering_radius_and_worst(best_points)
    
    points = best_points.copy()
    current_radius = best_radius
    
    # Simulated Annealing with Local Polish
    temp = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        current_radius, worst_simplex, worst_center, facet_radii = get_covering_radius_and_worst(points)
        
        if worst_simplex is None:
            break
            
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Local Polish: Optimize the worst facet and its neighbors
        if worst_simplex is not None:
            candidate = local_polish(points, worst_simplex, facet_radii)
            
            # Evaluate candidate
            cand_radius, _, _, _ = get_covering_radius_and_worst(candidate)
            
            # Accept if better, or with probability exp(-delta/temp) if worse
            delta = cand_radius - current_radius
            if delta < 0 or (temp > 1e-6 and np.random.rand() < np.exp(-delta / temp)):
                points = candidate
                current_radius = cand_radius
            
            temp *= cooling
            
            if temp < 1e-6:
                # If temperature is low, try a random restart of the local polish on a random facet?
                # Or just break if we've converged.
                # Let's try a few more iterations with very low temp to settle.
                pass

    # Final Polish on the best point
    best_radius, worst_simplex, _, facet_radii = get_covering_radius_and_worst(best_points)
    if worst_simplex is not None:
        for _ in range(10): # Do a few more polish steps on the best
            polished = local_polish(best_points, worst_simplex, facet_radii)
            pr, ws, _, fr = get_covering_radius_and_worst(polished)
            if pr < best_radius:
                best_points = polished
                best_radius = pr
                worst_simplex = ws
                facet_radii = fr
            else:
                break

    return best_points

def local_polish(points, worst_simplex, facet_radii):
    # Identify the set of facets to consider: the worst one and its neighbors
    # Neighbors share an edge or vertex.
    # For simplicity and speed, let's just optimize the 3 vertices of the worst facet
    # and maybe the vertices of facets sharing an edge.
    
    # Get hull to find neighbors? We don't have the hull object here, only indices.
    # We need to reconstruct the neighbor logic or just use the 3 vertices.
    # Optimizing just 3 vertices is fast.
    
    idxs = worst_simplex
    v0, v1, v2 = idxs
    
    # Define local objective
    def local_obj(x):
        # x is 9 values: 3 points
        new_pts = points.copy()
        new_pts[v0] = x[0:3]
        new_pts[v1] = x[3:6]
        new_pts[v2] = x[6:9]
        
        # Normalize
        new_pts = normalize(new_pts)
        
        # Compute radii for facets involving these vertices?
        # We don't have the list of all facets easily without the hull.
        # Let's just compute the radius of the worst facet and its 3 neighbors if we can find them.
        # Actually, we can just compute the global max radius? No, too slow for Nelder-Mead.
        
        # Approximation: Compute the circumradius of the facet defined by v0,v1,v2
        # and the 3 facets that share edges with it.
        # To find neighbors, we need the hull.
        # Let's pass the hull or compute it.
        # Computing hull for every local_obj call is too slow.
        
        # Alternative: Just minimize the circumradius of the worst facet itself.
        # This is a 3-point problem.
        
        p0, p1, p2 = new_pts[v0], new_pts[v1], new_pts[v2]
        
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            return np.pi
        n /= n_norm
        
        if n @ p0 < 0:
            n = -n
            
        dist = np.arccos(np.clip(n @ p0, -1, 1))
        
        # Add a penalty if the point moves too far? No, normalization handles it.
        # But we want to keep the other points fixed.
        
        return dist

    x0 = np.concatenate([points[v0], points[v1], points[v2]])
    
    # Use Nelder-Mead
    try:
        res = minimize(local_obj, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success:
            new_pts = points.copy()
            new_pts[v0] = res.x[0:3]
            new_pts[v1] = res.x[3:6]
            new_pts[v2] = res.x[6:9]
            return normalize(new_pts)
    except Exception:
        pass
        
    return points
