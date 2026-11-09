# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 10s safety margin


def _min_triangle_area(points):
    """Compute minimum area among all C(13,3)=286 triangles, vectorized."""
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return float(np.min(areas))


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # In 2D, volume = area
    except Exception:
        return 0.0


def _score(points):
    """Compute A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0:
        return 0.0
    return a_min / a_hull


def _make_points(R, r, offset_angle=0.0):
    """
    Construct 13 points: 7 on a regular heptagon of radius R,
    6 on a regular hexagon of radius r with rotational offset.
    """
    n_outer = 7
    n_inner = 6
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + offset_angle
    
    outer = np.column_stack([R * np.cos(angles_outer), R * np.sin(angles_outer)])
    inner = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    return np.vstack([outer, inner])


def _objective(params):
    """Objective to maximize: score. Returns negative for minimization."""
    R, r, offset = params
    if R <= 0 or r < 0 or r > R:
        return 0.0
    points = _make_points(R, r, offset)
    return -_score(points)


def _golden_section_1d(f, a, b, tol=1e-8, max_iter=200):
    """Golden section search for minimum of f on [a, b]."""
    gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
        if abs(b - a) < tol:
            break
    return (a + b) / 2


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points.
    
    Uses structured parameterization (7 outer + 6 inner) with golden section search.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    # Optimize over R, r, and offset angle
    # R is the outer radius, r is the inner radius
    # We can normalize R=1 and just optimize r and offset, then scale
    
    best_score = 0.0
    best_points = None
    
    # First: optimize r for a fixed offset, then optimize offset
    # Start with a coarse grid over offset, then refine
    
    # Try multiple offset angles
    offset_candidates = np.linspace(0, 2 * np.pi / 7, 20, endpoint=False)
    
    for offset in offset_candidates:
        if time.time() > _DEADLINE - 5:
            break
        
        # For this offset, optimize r using golden section search
        # r should be between 0 and 1 (with R=1)
        # The optimal r is likely around 0.3-0.7
        
        def obj_r(r):
            if r < 0 or r > 1:
                return 0.0
            points = _make_points(1.0, r, offset)
            return -_score(points)
        
        # Golden section search for r in [0.05, 0.95]
        r_opt = _golden_section_1d(obj_r, 0.05, 0.95, tol=1e-10, max_iter=150)
        
        points = _make_points(1.0, r_opt, offset)
        s = _score(points)
        
        if s > best_score:
            best_score = s
            best_points = points
    
    # Refine: do a local 2D optimization around the best found (r, offset)
    # Use a small grid refinement
    if best_points is not None:
        # Find the best offset and r from the coarse search
        # Redo with finer grid around best
        best_offset = None
        best_r = None
        best_s = 0.0
        
        for offset in offset_candidates:
            if time.time() > _DEADLINE - 5:
                break
            
            def obj_r(r):
                if r < 0 or r > 1:
                    return 0.0
                points = _make_points(1.0, r, offset)
                return -_score(points)
            
            r_opt = _golden_section_1d(obj_r, 0.05, 0.95, tol=1e-10, max_iter=150)
            points = _make_points(1.0, r_opt, offset)
            s = _score(points)
            
            if s > best_s:
                best_s = s
                best_r = r_opt
                best_offset = offset
        
        # Fine refinement around best_r and best_offset
        if best_r is not None and best_offset is not None:
            # Fine grid over r and offset near the optimum
            r_fine = np.linspace(max(0.01, best_r - 0.05), min(0.99, best_r + 0.05), 15)
            offset_fine = np.linspace(best_offset - 0.1, best_offset + 0.1, 15)
            
            for r_f in r_fine:
                if time.time() > _DEADLINE - 5:
                    break
                for off_f in offset_fine:
                    if time.time() > _DEADLINE - 5:
                        break
                    points = _make_points(1.0, r_f, off_f)
                    s = _score(points)
                    if s > best_s:
                        best_s = s
                        best_points = points
    
    # If we didn't find anything, fall back to a reasonable default
    if best_points is None:
        # Regular heptagon with 6 inner points
        best_points = _make_points(1.0, 0.4, 0.0)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = _make_points(1.0, 0.4, 0.0)
    
    return best_points


# EVOLVE-BLOCK-END
