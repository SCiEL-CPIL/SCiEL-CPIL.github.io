import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area_vec(points):
    """Exact minimum triangle area over all C(n,3) triples."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    m = float('inf')
    # Vectorized approach for speed
    # i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    xi, yi = x[i_idx[mask]], y[i_idx[mask]]
    xj, yj = x[j_idx[mask]], y[j_idx[mask]]
    xk, yk = x[k_idx[mask]], y[k_idx[mask]]
    areas = 0.5 * np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    return np.min(areas)

def _hull_area_fast(points):
    try:
        return ConvexHull(points).volume
    except Exception:
        return 0.0

def _score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 0.0
    ha = _hull_area_fast(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area_vec(points)
    if ma <= 0.0:
        return 0.0
    return ma / ha

def _regular_polygon(k, r=1.0, phase=0.0):
    a = phase + 2 * np.pi * np.arange(k) / k
    return r * np.column_stack([np.cos(a), np.sin(a)])

def _normalize_unit_area(points):
    """Scale so convex hull area = 1."""
    try:
        ha = ConvexHull(points).volume
        if ha > 1e-10:
            return points * np.sqrt(1.0 / ha)
    except Exception:
        pass
    xs, ys = points[:, 0], points[:, 1]
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        return points / np.sqrt(w * h)
    return points

def _objective_ellipse_gaps(params):
    """
    params: [a, b, d_1, ..., d_N]
    a, b: semi-axes of ellipse
    d_i: angular gaps, normalized to sum to 2*pi
    """
    if np.any(~np.isfinite(params)):
        return 1e10
    
    a, b = params[0], params[1]
    # Ensure a, b are positive and bounded
    if a < 0.1 or b < 0.1 or a > 10 or b > 10:
        return 1e10
        
    d = params[2:]
    d = np.maximum(d, 1e-6)
    d = d / np.sum(d) * 2 * np.pi
    
    thetas = np.cumsum(d)
    # Center the polygon? Not strictly necessary for area ratio, but good for symmetry.
    thetas = thetas - thetas[0] # Start at 0?
    
    x = a * np.cos(thetas)
    y = b * np.sin(thetas)
    pts = np.column_stack([x, y])
    
    # Hull Area (Shoelace)
    # Points are in angular order on the ellipse, so they form the convex hull.
    ha = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    
    if ha < 1e-12:
        return 1e10
        
    ma = _min_triangle_area_vec(pts)
    
    if ma <= 0:
        return 1e10
        
    ratio = ma / ha
    return -ratio

def _construct():
    t0 = time.time()
    deadline = t0 + 145.0
    best_score = 0.0
    best_pts = None
    
    # Initial best: Regular 14-gon
    reg_pts = _regular_polygon(N)
    best_score = _score(reg_pts)
    best_pts = reg_pts.copy()
    
    # Generate starts
    starts = []
    
    # 1. Regular polygon on circle (a=1, b=1, equal gaps)
    d_eq = np.ones(N)
    starts.append(np.concatenate([[1.0, 1.0], d_eq]))
    
    # 2. Ellipses with various aspect ratios
    for ratio in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]:
        a = np.sqrt(ratio)
        b = 1.0 / np.sqrt(ratio)
        starts.append(np.concatenate([[a, b], d_eq]))
        
    # 3. Perturbed gaps on circle
    rng = np.random.default_rng(42)
    for i in range(20):
        if time.time() > deadline - 20: break
        d = np.ones(N) + rng.normal(0, 0.1, N)
        d = np.abs(d)
        starts.append(np.concatenate([[1.0, 1.0], d]))
        
    # 4. Perturbed gaps on ellipses
    for i in range(20):
        if time.time() > deadline - 20: break
        ratio = rng.uniform(0.5, 2.0)
        a = np.sqrt(ratio)
        b = 1.0 / np.sqrt(ratio)
        d = np.ones(N) + rng.normal(0, 0.2, N)
        d = np.abs(d)
        starts.append(np.concatenate([[a, b], d]))

    # Optimize
    for idx, start in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(_objective_ellipse_gaps, start, method='Powell',
                           options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10})
            if res.success or np.isfinite(res.fun):
                # Reconstruct points
                a, b = res.x[0], res.x[1]
                d = res.x[2:]
                d = np.maximum(d, 1e-6)
                d = d / np.sum(d) * 2 * np.pi
                thetas = np.cumsum(d)
                x = a * np.cos(thetas)
                y = b * np.sin(thetas)
                pts = np.column_stack([x, y])
                
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

    return _normalize_unit_area(best_pts)

def construct():
    return _construct()
