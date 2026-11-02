"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
from scipy.optimize import minimize
import time

N = 14


def _triangle_areas(points):
    """Compute all triangle areas for n points. Returns array of size C(n,3)."""
    n = len(points)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                areas[idx] = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                idx += 1
    return areas


def _convex_hull_area(points):
    """Compute convex hull area using Andrew's monotone chain."""
    n = len(points)
    if n < 3:
        return 0.0
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) * 0.5


def _score(points):
    """Score = min triangle area / convex hull area. Higher is better."""
    try:
        areas = _triangle_areas(points)
        min_area = np.min(areas)
        hull_area = _convex_hull_area(points)
        if hull_area < 1e-12:
            return -1e10
        return min_area / hull_area
    except Exception:
        return -1e10


def _objective(x):
    """Negative score for minimization."""
    points = x.reshape(-1, 2)
    s = _score(points)
    return -s


def _make_initial_points(r_outer, r_inner, angle_offset, rotation):
    """Create 14 points: 7 on outer heptagon, 7 on inner heptagon with offset."""
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rotation
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + angle_offset + rotation
    
    outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    points = np.vstack([outer_pts, inner_pts])
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    configs = []
    
    # Two-ring heptagon configurations with varying parameters
    for r_outer in [1.0, 1.2, 1.5]:
        for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for angle_offset in [0, np.pi / 7, np.pi / 3.5, np.pi / 2]:
                for rotation in [0, np.pi / 14]:
                    configs.append((r_outer, r_inner, angle_offset, rotation))
    
    # Also try single-ring with perturbation
    for rotation in [0, np.pi / 14, np.pi / 7]:
        configs.append((1.0, 0.5, np.pi / 7, rotation))
    
    for config in configs:
        if time.time() > deadline:
            break
        
        r_outer, r_inner, angle_offset, rotation = config
        points = _make_initial_points(r_outer, r_inner, angle_offset, rotation)
        x0 = points.flatten()
        
        # Quick local optimization with Nelder-Mead
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if time.time() > deadline:
                break
            
            opt_points = result.x.reshape(-1, 2)
            s = _score(opt_points)
            
            if s > best_score:
                best_score = s
                best_points = opt_points.copy()
                
        except Exception:
            continue
    
    # If we haven't found anything, use the initial configuration
    if best_points is None:
        points = _make_initial_points(1.0, 0.5, np.pi / 7, 0)
        best_points = points
    
    return best_points
