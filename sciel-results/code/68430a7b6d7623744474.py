"""
Optimized covering of sphere with 50 points using facet-shrink and local polish.
Combines the structural insight of moving points towards circumcenters of worst facets
(inspiration) with robust initialization and time management (current program).
"""
import numpy as np
from scipy.spatial import ConvexHull
import time
import sys
import math

N = 50
_TIME_LIMIT = 140.0  # Safety margin before 150s kill
_DEADLINE = time.time() + _TIME_LIMIT


def _time_remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees using the convex hull facet circumradii.
    The covering radius is the maximum over all facets of the angular circumradius.
    """
    try:
        hull = ConvexHull(pts)
        max_cr = 0.0
        for simplex in hull.simplices:
            a, b, c = pts[simplex]
            
            # Compute normal to the triangle
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            
            # Ensure normal points outward (away from origin)
            if np.dot(n, a) < 0:
                n = -n
            
            # The circumcenter of the spherical triangle lies along the normal n.
            # We need to find the point on the sphere along n (or -n) that is equidistant 
            # from a, b, c. 
            # Actually, for a triangle on the sphere, the circumcenter in 3D space is 
            # not necessarily on the sphere. However, the "farthest point" in the Voronoi 
            # cell (which is a spherical polygon) is either a vertex (original points) 
            # or the circumcenter of a facet (if the circumcenter lies within the facet).
            # For the covering radius, we look for the point in the cell furthest from 
            # the vertices. This is the circumcenter of the spherical triangle formed 
            # by the three vertices, provided it lies inside the triangle.
            
            # Let's calculate the circumcenter of the spherical triangle.
            # The circumcenter C satisfies C.a = C.b = C.c = cos(alpha).
            # C is parallel to the cross product of (a-b) and (b-c) ? No, that's the normal.
            # Let's solve for C = x*a + y*b + z*c such that C is normalized and C.a=C.b=C.c.
            # Alternatively, the circumcenter vector is parallel to (a x b) x (b x c) ... 
            # Actually, a simpler geometric property: The circumcenter of the spherical triangle 
            # is the projection of the origin onto the plane of the triangle? No.
            
            # Let's use the property that the circumcenter C is equidistant from a,b,c.
            # C lies on the line passing through the origin and the circumcenter of the 
            # chord triangle? No.
            
            # Let's use the formula for the circumcenter of a spherical triangle.
            # Let A, B, C be the vertices.
            # The circumcenter U is given by U = (a sin(2A) + b sin(2B) + c sin(2C)) / |...|
            # This requires calculating angles, which is slow.
            
            # Alternative: The circumcenter of the triangle formed by vectors a, b, c on the sphere
            # is the unit vector u such that u.a = u.b = u.c.
            # This implies u is orthogonal to (a-b) and (b-c).
            # So u is parallel to (a-b) x (b-c).
            v = np.cross(a - b, b - c)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                # Degenerate triangle, check edge midpoints or just skip
                # If degenerate, the "circumcenter" is ill-defined, but the max distance
                # is likely determined by the longest edge midpoint if the triangle is flat.
                # For now, skip as it contributes 0 to max_cr if we assume non-degenerate.
                continue
                
            u = v / v_norm
            
            # We have two candidates: u and -u. We want the one "inside" the triangle.
            # The circumcenter is the one that has the same sign of dot product with the 
            # face normal? No.
            # The circumcenter C must satisfy C.a = C.b = C.c = cos(alpha).
            # Let's check u.a, u.b, u.c.
            # If u.a, u.b, u.c are all close to each other, u is a candidate.
            # If -u.a, -u.b, -u.c are all close to each other, -u is a candidate.
            
            da = np.dot(u, a)
            db = np.dot(u, b)
            dc = np.dot(u, c)
            
            # Check if u is the circumcenter. 
            # The circumcenter is the point on the sphere that is equidistant from a,b,c.
            # If the triangle is "small" (less than a hemisphere), the circumcenter is 
            # usually the one closer to the centroid of the triangle.
            centroid = (a + b + c) / 3.0
            centroid = _normalize(centroid)
            
            if np.dot(u, centroid) > 0:
                cc = u
            else:
                cc = -u
            
            # Calculate the angular radius from cc to the vertices
            cos_rad = np.dot(cc, a)
            cos_rad = np.clip(cos_rad, -1.0, 1.0)
            rad = np.arccos(cos_rad)
            
            max_cr = max(max_cr, rad)
            
        return np.degrees(max_cr)
    except Exception:
        return 180.0


def _facet_shrink_optimizer(pts, max_iter=500, initial_step=0.5, decay=0.99, beta_start=1.0, beta_end=50.0):
    """
    Optimizes the configuration by moving each point towards the weighted average
    of the circumcenters of its incident facets.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Calculate circumcenter and radius for each facet
        facet_data = []
        max_cr_global = 0.0
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v = np.cross(a - b, b - c)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            u = v / v_norm
            
            # Determine correct orientation
            centroid = (a + b + c) / 3.0
            if np.dot(u, centroid) < 0:
                u = -u
            
            # Calculate radius
            cos_rad = np.dot(u, a)
            cos_rad = np.clip(cos_rad, -1.0, 1.0)
            rad = np.arccos(cos_rad)
            
            max_cr_global = max(max_cr_global, rad)
            facet_data.append((simplex, u, rad))
            
        if not facet_data:
            break
            
        if max_cr_global < 1e-6:
            break

        # Interpolate beta
        progress = it / max_iter
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Accumulate forces
        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))
        
        for simplex, cc, rad in facet_data:
            # Weight by how much this facet contributes to the max radius
            # We want to move points to reduce the max radius.
            # If rad is close to max_cr_global, it's a "worst" facet.
            # We move the vertices of this facet towards cc?
            # Actually, moving the vertices towards the circumcenter of the facet 
            # shrinks the facet, reducing its circumradius.
            w = math.exp(beta * (rad - max_cr_global))
            
            for idx in simplex:
                weights_sum[idx] += w
                vec_sum[idx] += w * cc
                
        moved = False
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    
                    # Move point i towards target
                    # Project target to tangent plane at pts[i]
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vec = np.linalg.norm(vec_tan)
                    
                    if norm_vec > 1e-10:
                        pts[i] = pts[i] + step_size * vec_tan
                        moved = True
                        
        if not moved:
            break
            
        pts = _normalize(pts)
        
        # Decay step size
        step_size *= decay
        
        if step_size < 1e-8:
            break

    return pts


def _local_polish(pts, max_iter=100, tol=1e-10):
    """
    High-precision local optimization.
    Tries to reduce the max circumradius by fine-tuning points.
    """
    pts = _normalize(pts.copy())
    current_cr = _covering_radius(pts)
    best_pts = pts.copy()
    best_cr = current_cr
    
    step = 0.01
    for it in range(max_iter):
        if _time_remaining() < 3.0:
            break
            
        improved = False
        # Try moving each point in small random directions
        for i in range(N):
            for _ in range(5):
                d = np.random.randn(3)
                d /= np.linalg.norm(d)
                new_pts = pts.copy()
                new_pts[i] += step * d
                new_pts = _normalize(new_pts)
                
                new_cr = _covering_radius(new_pts)
                if new_cr < current_cr - 1e-12:
                    current_cr = new_cr
                    pts = new_pts
                    improved = True
                    if new_cr < best_cr:
                        best_cr = new_cr
                        best_pts = pts.copy()
                        
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
        else:
            # If improved, keep step or increase slightly? Keep it for stability.
            pass
            
    return best_pts


def _fibonacci_points():
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=None):
    """Random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _normalize(pts)


def construct():
    """Construct 50 points on unit sphere minimizing covering radius."""
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    np.random.seed(42)
    
    best_pts = None
    best_cr = float('inf')
    
    # Phase 1: Multiple starts with Facet Shrink Optimizer
    starts = [
        _fibonacci_points(),
        _random_points(seed=1),
        _random_points(seed=2),
        _random_points(seed=3),
        _random_points(seed=4),
    ]
    
    for s_idx, start in enumerate(starts):
        if _time_remaining() < 15.0:
            break
        
        # Run Facet Shrink Optimizer
        pts = _facet_shrink_optimizer(start, max_iter=200, initial_step=0.3, decay=0.995, beta_start=1.0, beta_end=30.0)
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    # Phase 2: Local Polish on the best configuration
    if best_pts is not None and _time_remaining() > 10.0:
        polished_pts = _local_polish(best_pts, max_iter=50)
        polished_cr = _covering_radius(polished_pts)
        
        if polished_cr < best_cr:
            best_cr = polished_cr
            best_pts = polished_pts
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_points()
        
    best_pts = _normalize(best_pts)
    
    # Final validation: check no coincident points, not in closed hemisphere
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10):
        best_pts = _fibonacci_points()
    
    # Check not in closed hemisphere: origin should be inside convex hull
    try:
        hull = ConvexHull(best_pts)
        inside = True
        for simplex in hull.simplices:
            tri = best_pts[simplex]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-15:
                n = n / n_norm
                if np.dot(n, tri[0]) < -1e-10:
                    inside = False
                    break
        if not inside:
            # Fallback to Fibonacci
            best_pts = _fibonacci_points()
    except Exception:
        best_pts = _fibonacci_points()
    
    return best_pts
