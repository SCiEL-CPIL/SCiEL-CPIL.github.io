"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _facet_circumradius(facet_pts):
    """Compute circumradius (in radians) of a spherical polygon given its vertices."""
    # The circumradius of a spherical polygon is the angular distance from the 
    # center of the circumscribed sphere to any vertex.
    # For a planar polygon on the unit sphere, we can compute the center as the 
    # normalized sum of vertices, then compute the max angular distance to vertices.
    if len(facet_pts) < 3:
        return 0.0
    
    # Compute centroid in 3D
    center_3d = np.mean(facet_pts, axis=0)
    center_norm = np.linalg.norm(center_3d)
    if center_norm < 1e-12:
        return np.pi / 2
    
    center = center_3d / center_norm
    
    # Compute angular distances from center to each vertex
    dots = np.sum(facet_pts * center, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    return np.max(angles)


def _covering_radius_deg(pts):
    """Compute covering radius in degrees using convex hull."""
    try:
        hull = ConvexHull(pts)
        max_cr = 0.0
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            cr = _facet_circumradius(facet_pts)
            if cr > max_cr:
                max_cr = cr
        return np.degrees(max_cr)
    except Exception:
        return np.pi


def _find_largest_facet(pts):
    """Find the facet with the largest circumradius."""
    try:
        hull = ConvexHull(pts)
        max_cr = -1.0
        best_simplex = None
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            cr = _facet_circumradius(facet_pts)
            if cr > max_cr:
                max_cr = cr
                best_simplex = simplex
        return best_simplex, max_cr
    except Exception:
        return None, 0.0


def _refine_facet(pts, simplex, step_size=0.05):
    """Move points of the largest facet toward the facet center to reduce circumradius."""
    pts = pts.copy()
    facet_pts = pts[simplex]
    
    # Compute facet center (normalized mean)
    center_3d = np.mean(facet_pts, axis=0)
    center_norm = np.linalg.norm(center_3d)
    if center_norm < 1e-12:
        return pts
    
    center = center_3d / center_norm
    
    # Move each point in the facet toward the center
    for idx in simplex:
        direction = center - pts[idx]
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-12:
            pts[idx] = pts[idx] + step_size * direction / dir_norm
    
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_cr = _covering_radius_deg(pts)
    
    # Adaptive step size
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Find the largest facet
        simplex, cr = _find_largest_facet(pts)
        if simplex is None:
            break
        
        # Try to refine this facet
        new_pts = _refine_facet(pts, simplex, step_size)
        
        # Check if it improved
        new_cr = _covering_radius_deg(new_pts)
        
        if new_cr < best_cr - 1e-10:
            best_pts = new_pts.copy()
            best_cr = new_cr
            pts = new_pts
            no_improve_count = 0
            # Increase step size slightly if improving
            step_size = min(step_size * 1.1, 0.15)
        else:
            # No improvement, decrease step size
            step_size = max(step_size * 0.8, 0.01)
            no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                # Try a random restart with small perturbation
                noise = np.random.randn(N, 3) * 0.01
                pts = _normalize(pts + noise)
                new_cr = _covering_radius_deg(pts)
                if new_cr < best_cr:
                    best_pts = pts.copy()
                    best_cr = new_cr
                no_improve_count = 0
                step_size = 0.05
    
    # Final check
    final_cr = _covering_radius_deg(best_pts)
    if final_cr < best_cr:
        best_cr = final_cr
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
