import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
import time

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius for points on the unit sphere.
    The covering radius is the maximum distance from any point on the sphere
    to the nearest point in 'points'.
    This is equal to the maximum circumradius of the spherical Voronoi cells.
    """
    try:
        vor = SphericalVoronoi(points)
        # Voronoi vertices are the circumcenters of the Delaunay triangles
        # The distance from a Voronoi vertex to its corresponding sites is the circumradius
        # We need the max over all Voronoi vertices of the distance to the nearest site?
        # Actually, each Voronoi vertex is equidistant to 3 sites.
        # The distance to any of those 3 sites is the same.
        # So we just need the norm of the Voronoi vertices?
        # No, the Voronoi vertices are on the unit sphere.
        # The angular distance is the angle between the Voronoi vertex and the site.
        # Let v be a Voronoi vertex. Let s1, s2, s3 be the sites.
        # dist(v, s1) = arccos(dot(v, s1)).
        # Since v is a vertex, it is the circumcenter of the triangle s1,s2,s3.
        # So dot(v, s1) = dot(v, s2) = dot(v, s3).
        # We want the max of these distances.
        
        # vor.vertices are the points on the sphere.
        # We need to associate each vertex with a site to compute the distance.
        # vor.point_region gives the region for each point.
        # But we need the distance from the vertex to the point.
        
        # A simpler way:
        # The covering radius is the max over all Voronoi vertices v of min_i dist(v, p_i).
        # But for a Voronoi vertex v, the min_i dist(v, p_i) is the distance to the sites that define it.
        
        # Let's just compute the distance from each Voronoi vertex to all points and take the min?
        # That is O(V * N). V is ~2N. N=130. 260 * 130 = 33800 ops. Very fast.
        
        vertices = vor.vertices
        # Compute distances from each vertex to all points
        # dot product
        dots = vertices @ points.T
        # Clip to [-1, 1]
        dots = np.clip(dots, -1.0, 1.0)
        # Distance is arccos(dot)
        # We want min distance for each vertex
        min_dots = np.max(dots, axis=1) # max dot is min distance
        min_dists = np.arccos(min_dots)
        
        return np.max(min_dists)
        
    except Exception:
        # If Voronoi fails, fallback to a slower but robust method
        # Sample many points and find min dist
        # Or return a large value
        return np.pi / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # 2. Lloyd's Algorithm (Spherical K-Means)
    # Iterate to find local optimum for sum-of-squares, which is a good proxy for covering
    for k in range(50):
        if time.time() > deadline - 30.0:
            break
            
        try:
            vor = SphericalVoronoi(points)
            # Compute centroids
            new_points = np.zeros_like(points)
            for i in range(N):
                region = vor.regions[i]
                if not region:
                    new_points[i] = points[i]
                    continue
                verts = vor.vertices[region]
                # Centroid approx: normalized sum of vertices
                c = np.sum(verts, axis=0)
                norm = np.linalg.norm(c)
                if norm < 1e-10:
                    new_points[i] = points[i]
                else:
                    new_points[i] = c / norm
            
            # Check for significant movement
            diff = np.linalg.norm(new_points - points, axis=1)
            if np.max(diff) < 1e-6:
                break
                
            points = new_points
            
            # Update best if improved
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
        except Exception:
            break
            
    # 3. Polish with Nelder-Mead on the covering radius
    # We need a function to evaluate the radius for a given set of points
    # Nelder-Mead works on a flat vector of 3N coordinates.
    
    def objective(flat_points):
        if time.time() > deadline:
            return 10.0 # Large penalty
        pts = flat_points.reshape(N, 3)
        # Normalize points to ensure they are on the sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        return _compute_covering_radius(pts)
        
    initial_flat = points.flatten()
    
    # Use Nelder-Mead
    # maxiter needs to be reasonable.
    # Each eval is ~ms. 1000 evals is fine.
    try:
        res = minimize(objective, initial_flat, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if res.fun < best_radius:
            best_radius = res.fun
            best_flat = res.x
            norms = np.linalg.norm(best_flat.reshape(N, 3), axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            best_points = best_flat.reshape(N, 3) / norms
    except Exception:
        pass

    # Final check and return
    # Ensure points are normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points
