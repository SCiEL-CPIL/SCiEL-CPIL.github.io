"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(pts):
    """
    Compute the covering radius of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This is equivalent to the largest circumradius of the
    facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(pts)
        max_rad = 0.0
        for simplex in hull.simplices:
            # Get the 3 points of the facet
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            # Compute circumcenter and radius of the spherical triangle
            # The circumcenter of a spherical triangle is the point equidistant from all three vertices
            # We can find it by finding the intersection of the perpendicular bisectors of the arcs
            # Or more simply: the circumcenter is along the normal to the plane of the triangle
            # But for a spherical triangle, the circumcenter is the point on the sphere 
            # equidistant from a, b, c.
            
            # The circumcenter O satisfies: O·a = O·b = O·c (since |O|=|a|=|b|=|c|=1)
            # This means O is perpendicular to (a-b) and (a-c), so O is along the cross product (b-a)×(c-a)
            # But we need the one on the sphere that is the circumcenter of the triangle.
            
            # Actually, for a spherical triangle, the circumcenter is the point on the sphere
            # that is equidistant from all three vertices. This is the intersection of the sphere
            # with the line through the origin in the direction of the normal to the plane of the triangle.
            
            # Let n = (b-a) × (c-a), normalized
            # Then the circumcenter is either n or -n, whichever gives the smaller circumradius
            # (i.e., whichever is on the same side as the triangle's interior)
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                # Degenerate triangle
                continue
            n = n / n_norm
            
            # The circumcenter is at distance R from each vertex, where cos(R) = n·a (or -n·a)
            # We need to choose the sign of n such that the circumcenter is on the correct side
            # (the side where the triangle's interior is)
            
            # The triangle's centroid is (a+b+c)/3, normalized
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                continue
            centroid = centroid / centroid_norm
            
            # Choose n such that n·centroid > 0 (circumcenter on the same side as centroid)
            if np.dot(n, centroid) < 0:
                n = -n
            
            # The circumradius R satisfies cos(R) = n·a
            cos_R = np.dot(n, a)
            # Clamp for numerical safety
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            
            if R > max_rad:
                max_rad = R
        
        return max_rad
    except Exception:
        return np.pi  # Fallback: worst case


def _covering_radius_degrees(pts):
    return np.degrees(_covering_radius(pts))


def _objective(x, pts_fixed):
    """
    Objective function for optimization: minimize the covering radius.
    x is a 3D vector representing one point to optimize.
    pts_fixed is the array of other fixed points.
    """
    x = np.array(x)
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        return np.pi
    x = x / norm
    
    # Combine with fixed points
    all_pts = np.vstack([x, pts_fixed])
    return _covering_radius(all_pts)


def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    best_pts = pts.copy()
    best_rad = _covering_radius(pts)
    
    # Iterative local optimization: move each point to minimize the covering radius
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            # Save current point
            orig_pt = pts[idx].copy()
            
            # Fix all other points
            pts_fixed = np.delete(pts, idx, axis=0)
            
            # Optimize this point
            best_local = orig_pt
            best_local_rad = _covering_radius(pts)
            
            # Try a few random perturbations
            for trial in range(10):
                if time.time() > deadline:
                    break
                
                # Random perturbation
                noise = np.random.randn(3) * 0.1
                candidate = orig_pt + noise
                candidate = candidate / np.linalg.norm(candidate)
                
                # Replace and evaluate
                pts[idx] = candidate
                rad = _covering_radius(pts)
                
                if rad < best_local_rad:
                    best_local_rad = rad
                    best_local = candidate.copy()
            
            pts[idx] = best_local
            
            if best_local_rad < best_rad - 1e-10:
                best_rad = best_local_rad
                best_pts = pts.copy()
                improved = True
        
        if not improved:
            break
    
    return best_pts
# EVOLVE-BLOCK-END
