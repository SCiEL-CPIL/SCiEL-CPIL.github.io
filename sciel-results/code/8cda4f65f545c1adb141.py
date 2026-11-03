"""New program using Lloyd's Algorithm (Voronoi Relaxation) for sphere covering."""
import numpy as np
from scipy.spatial import ConvexHull

N = 50

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in degrees using Convex Hull facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_radius_deg = 0.0
    for simplex in hull.simplices:
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        
        a = p1 - p0
        b = p2 - p0
        aa = np.dot(a, a)
        bb = np.dot(b, b)
        ab = np.dot(a, b)
        
        denom = aa * bb - ab * ab
        if abs(denom) < 1e-15:
            continue
            
        s = (aa * bb - ab * ab) / (2.0 * denom)
        t = (aa * ab - aa * aa) / (2.0 * denom)
        
        circumcenter_2d = p0 + s * a + t * b
        R = np.linalg.norm(circumcenter_2d - p0)
        C_norm = np.linalg.norm(circumcenter_2d)
        
        if C_norm < 1e-15:
            theta = np.pi / 2.0
        else:
            cos_theta = (C_norm * C_norm + 1.0 - R * R) / (2.0 * C_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
        
        radius_deg = np.degrees(theta)
        if radius_deg > max_radius_deg:
            max_radius_deg = radius_deg
            
    return max_radius_deg

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    """
    Lloyd's Algorithm (Voronoi Relaxation) to minimize covering radius.
    """
    # Start with Fibonacci sphere
    pts = _normalize(_fibonacci_sphere(N))
    
    best_pts = pts.copy()
    best_radius = _covering_radius_exact(best_pts)
    
    max_iter = 1000
    tol = 1e-10
    
    for k in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            # If hull fails, break
            break
            
        # Map point index to list of incident facet circumcenters
        # Each facet is a triangle of 3 points.
        # The circumcenter of the facet is a vertex of the Voronoi cell for each of the 3 points.
        
        # 1. Compute circumcenter for each facet
        n_facets = len(hull.simplices)
        circumcenters = np.zeros((n_facets, 3))
        
        for idx, simplex in enumerate(hull.simplices):
            p0 = pts[simplex[0]]
            p1 = pts[simplex[1]]
            p2 = pts[simplex[2]]
            
            a = p1 - p0
            b = p2 - p0
            aa = np.dot(a, a)
            bb = np.dot(b, b)
            ab = np.dot(a, b)
            
            denom = aa * bb - ab * ab
            if abs(denom) < 1e-15:
                circumcenters[idx] = (p0 + p1 + p2) / 3.0 # Fallback
                continue
                
            s = (aa * bb - ab * ab) / (2.0 * denom)
            t = (aa * ab - aa * aa) / (2.0 * denom)
            
            cc = p0 + s * a + t * b
            # Normalize circumcenter to be on the sphere? 
            # The Voronoi vertex is the intersection of the plane perpendicular to the edge bisectors.
            # The vector from origin to circumcenter is the direction of the Voronoi vertex.
            # We just need the direction.
            norm_cc = np.linalg.norm(cc)
            if norm_cc > 1e-10:
                circumcenters[idx] = cc / norm_cc
            else:
                circumcenters[idx] = (p0 + p1 + p2) / 3.0
                circumcenters[idx] /= np.linalg.norm(circumcenters[idx])

        # 2. For each point, sum the circumcenters of its incident facets
        new_pts = np.zeros_like(pts)
        
        # Create a mapping from point index to list of facet indices
        point_to_facets = [[] for _ in range(N)]
        for idx, simplex in enumerate(hull.simplices):
            for p_idx in simplex:
                point_to_facets[p_idx].append(idx)
                
        for i in range(N):
            if not point_to_facets[i]:
                # Point is not on hull? Should not happen for convex hull of points on sphere
                new_pts[i] = pts[i]
                continue
                
            # Sum the circumcenters
            sum_cc = np.sum(circumcenters[point_to_facets[i]], axis=0)
            
            # Normalize
            norm_sum = np.linalg.norm(sum_cc)
            if norm_sum > 1e-10:
                new_pts[i] = sum_cc / norm_sum
            else:
                new_pts[i] = pts[i]
        
        # 3. Check convergence
        diff = np.linalg.norm(new_pts - pts, axis=1)
        max_diff = np.max(diff)
        
        # Update points
        pts = new_pts
        
        # Evaluate
        current_radius = _covering_radius_exact(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
            
        if max_diff < tol:
            break
            
    return best_pts
