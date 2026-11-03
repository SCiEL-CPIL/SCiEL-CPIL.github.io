import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            # Compute the normal to the plane defined by p1, p2, p3
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            # Ensure the normal points outward (away from origin)
            if np.dot(c, p1) < 0:
                c = -c
            
            # The angular distance from the circumcenter c to the vertices is the circumradius.
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return np.pi / 2
        
    return np.degrees(max_radius)

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _lloyd_iteration(points):
    """
    Perform one iteration of Lloyd's algorithm.
    1. Compute convex hull.
    2. For each point, find the circumcenters of the triangles incident to it.
    3. Move the point to the normalized sum of these circumcenters.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Map point index to list of circumcenters of incident facets
        # We need to identify which facets contain which points.
        # hull.simplices is an array of shape (M, 3) containing indices of vertices.
        
        circumcenters = []
        # We also need to know which circumcenter belongs to which point.
        # A circumcenter of a triangle (i, j, k) is a vertex of the Voronoi cell of i, j, and k.
        
        # Initialize sum of circumcenters for each point
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        for tri in facets:
            i, j, k = tri
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            if np.dot(c, p1) < 0:
                c = -c
            
            # This circumcenter c is a vertex of the Voronoi cell for i, j, and k
            sums[i] += c
            sums[j] += c
            sums[k] += c
            
            counts[i] += 1
            counts[j] += 1
            counts[k] += 1
        
        new_points = points.copy()
        for i in range(N):
            if counts[i] > 0:
                # Move to the normalized sum of the circumcenters
                # This approximates the centroid of the Voronoi cell
                new_points[i] = sums[i]
                norm = np.linalg.norm(new_points[i])
                if norm > 1e-12:
                    new_points[i] /= norm
                else:
                    # Fallback if sum is zero (should not happen for well-distributed points)
                    new_points[i] = points[i]
            
        return new_points
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Run Lloyd's algorithm
    # We can also mix in some random restarts or perturbations if it stalls, 
    # but pure Lloyd's is often strong.
    
    # To handle potential local minima, we can run Lloyd's, and if improvement stalls,
    # apply a small random perturbation to a few points and continue.
    
    last_improvement_time = start_time
    stall_count = 0
    max_stall = 20 # Number of iterations without improvement before perturbing
    
    while time.time() < deadline:
        new_points = _lloyd_iteration(points)
        
        # Ensure points are on the sphere
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points.copy()
            last_improvement_time = time.time()
            stall_count = 0
        else:
            # If no improvement, keep old points? Or accept if close?
            # In Lloyd's, the objective usually decreases monotonically or oscillates slightly.
            # If it increases, it might be a bad step.
            # However, sometimes a slight increase in one step leads to a better global minimum.
            # For simplicity and safety, we only accept if it improves the best found.
            # But we should update 'points' to 'new_points' to continue the trajectory?
            # If we don't, we are stuck.
            # Let's accept the step if it's not significantly worse, or just always accept
            # and track the best.
            
            # Actually, standard Lloyd's moves to the centroid. The covering radius should generally decrease.
            # If it increases, it might be due to numerical issues or a bad configuration.
            # Let's try to accept the step anyway to escape local minima, but keep the best.
            points = new_points.copy()
            stall_count += 1
            
            # If we've been stalled for a while, add noise to some points
            if stall_count > max_stall:
                # Pick a random subset of points to perturb
                num_perturb = N // 5
                indices = np.random.choice(N, num_perturb, replace=False)
                noise = np.random.normal(0, 0.01, (num_perturb, 3))
                points[indices] += noise
                points[indices] /= np.linalg.norm(points[indices], axis=1, keepdims=True)
                stall_count = 0
                # Re-evaluate after perturbation? The next loop will do it.
                
    return best_points
