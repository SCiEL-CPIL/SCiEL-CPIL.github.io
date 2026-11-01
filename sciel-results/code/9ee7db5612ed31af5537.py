import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _normalize(v):
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1e-10
    return v / norms[:, np.newaxis]

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n):
    pts = np.random.randn(n, 3)
    return _normalize(pts)

def _get_worst_facet(pts):
    """
    Returns the index of the facet with the largest 'circumradius' approximation
    (max distance from centroid to vertices) and the centroid itself.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, 0.0
    
    max_dist = 0.0
    worst_facet_idx = -1
    worst_centroid = None
    
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        # Centroid approximation
        centroid = np.mean(facet_pts, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-10:
            continue
        centroid = centroid / centroid_norm
        
        # Distances from centroid to vertices
        # Use 1 - dot product as a proxy for angular distance (monotonic for small angles)
        dots = np.dot(facet_pts, centroid)
        dists = 1.0 - dots
        d_max = np.max(dists)
        
        if d_max > max_dist:
            max_dist = d_max
            worst_facet_idx = simplex
            worst_centroid = centroid
            
    return worst_facet_idx, worst_centroid, max_dist

def _local_search(pts, max_time=10.0):
    """
    Perform local search by moving vertices of the worst facet towards its centroid.
    """
    start_time = time.time()
    step_size = 0.05
    pts = pts.copy()
    
    while time.time() - start_time < max_time:
        facet_idx, centroid, max_dist = _get_worst_facet(pts)
        
        if facet_idx is None:
            break
            
        # Move vertices of the worst facet towards the centroid
        # This reduces the size of the facet
        for i in facet_idx:
            # Gradient direction: towards centroid
            # p_new = p + step * (centroid - p)
            # But we must stay on the sphere.
            # Simple projection: p_new = normalize(p + step * (centroid - p))
            delta = centroid - pts[i]
            pts[i] = pts[i] + step_size * delta
            pts[i] = pts[i] / np.linalg.norm(pts[i])
            
        # Occasionally, try to move a random point to escape local minima?
        # Or use adaptive step size.
        if max_dist < 0.05: # ~3 degrees, quite good
             step_size *= 0.9
             
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_pts = None
    best_score = float('inf')
    
    # Multi-start
    # Start 1: Fibonacci
    pts = _fibonacci_spiral(N)
    pts = _local_search(pts, max_time=20.0)
    
    # Evaluate score (approximate covering radius)
    # We need a better score function.
    # The actual score is the max angular distance to nearest point.
    # We can approximate this by sampling points on the sphere?
    # Or use the convex hull facet circumradii.
    
    def _score(pts):
        try:
            hull = ConvexHull(pts)
            max_r = 0
            for simplex in hull.simplices:
                facet_pts = pts[simplex]
                centroid = np.mean(facet_pts, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                dots = np.dot(facet_pts, centroid)
                # Angle = arccos(dot)
                # Max angle for this facet
                min_dot = np.min(dots)
                angle = np.arccos(np.clip(min_dot, -1, 1))
                if angle > max_r:
                    max_r = angle
            return np.degrees(max_r)
        except Exception:
            return float('inf')
            
    score = _score(pts)
    if score < best_score:
        best_score = score
        best_pts = pts.copy()
        
    # More random starts
    while time.time() < deadline:
        pts = _random_points(N)
        pts = _local_search(pts, max_time=10.0)
        score = _score(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
            
    return best_pts
