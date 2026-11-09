import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_circumcenters(points, simplices):
    """
    Compute circumcenters for all simplices in the convex hull.
    Returns an array of circumcenters (N_facets, 3).
    """
    if len(simplices) == 0:
        return np.zeros((0, 3))
        
    v1 = points[simplices[:, 0]]
    v2 = points[simplices[:, 1]]
    v3 = points[simplices[:, 2]]
    
    # Normal to the plane of the triangle
    n = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(n, axis=1)
    
    # Avoid division by zero
    valid = norms > 1e-10
    circumcenters = np.zeros((len(simplices), 3))
    
    if np.any(valid):
        n_valid = n[valid] / norms[valid, None]
        # Determine correct sign: circumcenter should be on the same side as the centroid of the triangle
        centroids = (v1[valid] + v2[valid] + v3[valid]) / 3.0
        dots = np.sum(n_valid * centroids, axis=1)
        signs = np.sign(dots)
        # If sign is 0, it's degenerate, but valid mask should have caught most.
        # If sign is negative, flip.
        n_valid *= signs[:, None]
        circumcenters[valid] = n_valid
        
    return circumcenters

def _compute_lloyd_targets(points):
    """
    Compute the Lloyd's algorithm targets (spherical centroids of Voronoi cells)
    and the current covering radius.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
        if len(simplices) == 0:
            return points.copy(), np.pi
            
        circumcenters = _compute_circumcenters(points, simplices)
        
        # Calculate covering radius: max angle between circumcenter and its vertices
        # For each simplex, the circumcenter is equidistant from v1, v2, v3.
        # The radius is the angle between circumcenter and any vertex.
        v1 = points[simplices[:, 0]]
        dots = np.sum(circumcenters * v1, axis=1)
        # Clip to [-1, 1] for arccos
        dots = np.clip(dots, -1.0, 1.0)
        radii = np.arccos(dots)
        
        if len(radii) == 0:
            max_radius = np.pi
        else:
            max_radius = np.max(radii)
            
        # Compute Voronoi cell centroids for each point
        new_points = np.zeros_like(points)
        
        # Map point index to list of circumcenter indices
        accum = np.zeros_like(points)
        
        for i, simplex in enumerate(simplices):
            cc = circumcenters[i]
            for idx in simplex:
                accum[idx] += cc
                
        norms = np.linalg.norm(accum, axis=1)
        valid = norms > 1e-10
        new_points[valid] = accum[valid] / norms[valid, None]
        
        # If a point has no facets, keep it
        new_points[~valid] = points[~valid]
        
        return new_points, max_radius

    except Exception:
        return points.copy(), np.pi

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    try:
        best_radius, _ = _compute_lloyd_targets(best_points)
    except Exception:
        best_radius = np.pi
        
    # Optimization parameters
    alpha = 0.5 # Relaxation factor for moving towards centroid
    
    stagnation_counter = 0
    
    while time.time() < deadline:
        try:
            targets, current_radius = _compute_lloyd_targets(points)
            
            if current_radius < best_radius - 1e-12:
                best_radius = current_radius
                best_points = points.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # Move towards targets
            diffs = targets - points
            new_points = points + alpha * diffs
            norms = np.linalg.norm(new_points, axis=1)
            new_points = new_points / norms[:, None]
            
            # Check for coincident points or invalid states
            if not np.all(np.isfinite(new_points)):
                break
                
            points = new_points
            
            # If stagnating, try a random perturbation to escape local minimum
            if stagnation_counter > 10:
                # Small random perturbation
                noise = np.random.normal(0, 0.1, size=points.shape)
                points = points + noise
                norms = np.linalg.norm(points, axis=1)
                points = points / norms[:, None]
                stagnation_counter = 0
                alpha = 0.5 # Reset alpha
                
        except Exception:
            # If an error occurs, break the loop and return best
            break
            
    return best_points
