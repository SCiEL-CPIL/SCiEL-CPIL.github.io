"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import numpy as np
from scipy.optimize import minimize
import time

N = 14


def _min_triangle_area(points):
    """Compute the minimum area over all C(14,3)=364 triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Area of convex hull. For points in convex position, this is the polygon area."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: if all points are in convex position, compute polygon area
        # Sort by angle from centroid
        cx, cy = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        order = np.argsort(angles)
        pts = points[order]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
        return abs(area) * 0.5


def _score(points):
    """Min triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _from_angles(angles):
    """Convert sorted angles to points on unit circle, then scale to unit area."""
    angles = np.sort(angles)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Compute area of this polygon and scale to unit area
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
    area = abs(area) * 0.5
    if area < 1e-15:
        return pts
    scale = np.sqrt(1.0 / area)
    return pts * scale


def _objective(angles):
    """Negative score (we minimize)."""
    pts = _from_angles(angles)
    return -_score(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    
    # Strategy: parameterize by N angles (convex position on circle),
    # optimize with Nelder-Mead from multiple random starts
    
    rng = np.random.RandomState(42)
    
    # Initial best: regular polygon
    regular_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular_pts = _from_angles(regular_angles)
    best_score = _score(regular_pts)
    best_points = regular_pts.copy()
    
    # Multiple random restarts with Nelder-Mead
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random angles (sorted)
        if restart == 0:
            angles0 = regular_angles.copy()
        else:
            # Random perturbation of regular polygon, or fully random
            if rng.random() < 0.5:
                angles0 = regular_angles + rng.randn(N) * 0.1
            else:
                angles0 = rng.uniform(0, 2 * np.pi, N)
                angles0 = np.sort(angles0)
        
        angles0 = np.sort(angles0)
        
        # Nelder-Mead optimization
        try:
            result = minimize(
                _objective,
                angles0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        opt_angles = np.sort(result.x)
        opt_pts = _from_angles(opt_angles)
        opt_score = _score(opt_pts)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
        
        # If we have good time left, do a second round of refinement
        if time.time() > deadline - 30:
            break
    
    # Additional refinement: coordinate descent on angles
    if best_points is not None and time.time() < deadline - 10:
        angles = np.arctan2(best_points[:, 1], best_points[:, 0])
        angles = np.sort(angles)
        
        step = 0.01
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                for delta in [step, -step]:
                    trial = angles.copy()
                    trial[i] += delta
                    trial = np.sort(trial)
                    pts = _from_angles(trial)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
                        angles = trial
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
    
    return best_points
