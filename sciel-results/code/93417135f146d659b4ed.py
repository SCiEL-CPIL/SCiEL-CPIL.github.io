import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    i, j, k = i.T.ravel(), j.T.ravel(), k.T.ravel()
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Return the minimum triangle area, or 0 if degenerate."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        areas = _triangle_areas(points)
        return float(np.min(areas))
    except Exception:
        return 0.0

def _fast_local_optimize(points, max_rounds=50, tol=1e-10):
    """Fast coordinate descent: for each point, try moving in 8 directions."""
    pts = points.copy().astype(float)
    n = len(pts)
    best_score = _score(pts)
    
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [0.7071, 0.7071], [-0.7071, 0.7071], [0.7071, -0.7071], [-0.7071, -0.7071]
    ])
    
    for rnd in range(max_rounds):
        improved = False
        for i in range(n):
            for d in directions:
                for step in [0.05, 0.02, 0.01, 0.005]:
                    new_pts = pts.copy()
                    new_pts[i] += d * step
                    s = _score(new_pts)
                    if s > best_score + tol:
                        pts = new_pts
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return pts, best_score

def _generate_initial_points(method='polygon', seed=42):
    """Generate diverse initial point sets."""
    n = 13
    rng = np.random.default_rng(seed)
    
    if method == 'polygon':
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    elif method == 'grid':
        # 4x4 grid minus one corner, slightly perturbed
        xs = np.linspace(-1, 1, 4)
        ys = np.linspace(-1, 1, 4)
        pts = np.array([[x, y] for x in xs for y in ys])
        pts = pts[:13]
        pts += rng.normal(0, 0.05, size=(13, 2))
        return pts
    elif method == 'random':
        pts = rng.uniform(-1, 1, size=(13, 2))
        return pts
    elif method == 'mixed':
        # Mix of polygon and random
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        pts += rng.normal(0, 0.1, size=(13, 2))
        return pts
    else:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 580  # Safety margin
    
    # Try multiple initializations
    best_points = None
    best_score = 0.0
    
    methods = ['polygon', 'grid', 'mixed', 'random']
    seeds = [42, 123, 456, 789, 101, 202, 303]
    
    for method in methods:
        for seed in seeds:
            if time.time() > deadline - 60:
                break
            
            init_pts = _generate_initial_points(method, seed)
            
            # Fast local optimization
            opt_pts, opt_score = _fast_local_optimize(init_pts, max_rounds=30)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        
        if time.time() > deadline - 60:
            break
    
    # If we have a good result, do more refined optimization
    if best_points is not None and time.time() < deadline - 30:
        # Try more rounds of local optimization
        refined_pts, refined_score = _fast_local_optimize(best_points, max_rounds=80)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Final fallback
    if best_points is None or _score(best_points) <= 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
