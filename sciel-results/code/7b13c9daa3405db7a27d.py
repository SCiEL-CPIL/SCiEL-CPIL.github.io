import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate indices for all combinations i < j < k
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i = i[mask].flatten()
    j = j[mask].flatten()
    k = k[mask].flatten()
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _neg_score(points_flat):
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring_config(n1, n2, r, theta):
    """Generate two-ring configuration."""
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + theta
    
    p1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r * np.column_stack([np.cos(angles2), np.sin(angles2)])
    
    return np.vstack([p1, p2])

def _local_optimize(points, time_limit=20.0):
    """Nelder-Mead refinement."""
    start_time = time.time()
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() - start_time < time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
        
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # 1. Initialize with regular polygon
    init_pts = _regular_polygon(N, 1.0)
    s, p = _local_optimize(init_pts, time_limit=10.0)
    if s > best_overall_score:
        best_overall_score = s
        best_overall_points = p
    
    # 2. Try Two-Ring configurations with local optimization
    # Search over r and theta
    for r in [0.3, 0.5, 0.7, 0.9]:
        for theta in [0.0, np.pi/N, np.pi/(2*N)]:
            if time.time() > deadline - 30:
                break
            pts = _two_ring_config(7, 7, r, theta)
            s, p = _local_optimize(pts, time_limit=10.0)
            if s > best_overall_score:
                best_overall_score = s
                best_overall_points = p
    
    # 3. Differential Evolution on 2D coordinates
    # This is the main structural change: robust global search
    if time.time() < deadline - 60:
        bounds = [(-2, 2)] * (2 * N)
        
        # Use a smaller population to fit in time, but more iterations
        # Or use the time limit feature of DE
        try:
            res = differential_evolution(
                _neg_score,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=False,
                timeout=45  # Leave 15s for polish
            )
            if time.time() < deadline - 10:
                pts = res.x.reshape(-1, 2)
                s, p = _local_optimize(pts, time_limit=10.0)
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall_points = p
        except Exception:
            pass

    # 4. Final Polish
    if time.time() < deadline - 5 and best_overall_points is not None:
        s, p = _local_optimize(best_overall_points, time_limit=10.0)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p

    if best_overall_points is None:
        best_overall_points = init_pts

    return best_overall_points
