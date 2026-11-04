"""Heilbronn convex regions n=14 — superellipse (Lamé curve) with arc-length uniform distribution."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar, minimize

N = 14

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using pairwise cross products."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n - 2):
        a = points[i]
        for j in range(i + 1, n - 1):
            b = points[j]
            dk = points[j+1:n] - a
            db = b - a
            crosses = db[0] * dk[:, 1] - db[1] * dk[:, 0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        n = len(points)
        if n < 3:
            return 0.0
        cx, cy = points[:, 0].mean(), points[:, 1].mean()
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        order = np.argsort(angles)
        hpts = points[order]
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hpts[i][0] * hpts[j][1] - hpts[j][0] * hpts[i][1]
        return abs(area) * 0.5

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _superellipse_points(n_exp, aspect, n_pts=14):
    """
    Generate n_pts points uniformly distributed by arc length on a superellipse.
    Superellipse: |x/a|^n + |y/b|^n = 1
    We parameterize by angle theta: x = a * sign(cos(theta)) * |cos(theta)|^(2/n)
                                      y = b * sign(sin(theta)) * |sin(theta)|^(2/n)
    But we want uniform arc length, so we sample by arc length.
    """
    a = 1.0
    b = aspect
    
    # Parameterize by theta in [0, 2*pi)
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    x = a * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** (2.0 / n_exp)
    y = b * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** (2.0 / n_exp)
    
    # Compute cumulative arc length
    dx = np.diff(x, axis=0, append=x[:1])
    dy = np.diff(y, axis=0, append=y[:1])
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)
    s_total = s[-1]
    s = s - s[0]  # start from 0
    
    # Sample uniformly by arc length
    target_s = np.linspace(0, s_total, n_pts, endpoint=False)
    
    # Interpolate to get points at uniform arc length
    x_pts = np.interp(target_s, s, x, period=s_total)
    y_pts = np.interp(target_s, s, y, period=s_total)
    
    return np.column_stack((x_pts, y_pts))

def _score_superellipse(n_exp, aspect):
    pts = _superellipse_points(n_exp, aspect)
    return _score(pts)

def _local_optimize_free(points, time_limit=15.0):
    """Local optimization in full 2D space."""
    start_time = time.time()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() < start_time + time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
    
    # One restart with small perturbation
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        perturb_scale = 0.03
        x_perturbed = best_pts.flatten() + perturb_scale * np.random.randn(len(best_pts.flatten()))
        try:
            res2 = minimize(
                obj, x_perturbed, method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if time.time() < start_time + time_limit:
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_pts = res2.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    return best_score, best_pts

def _optimize_superellipse(time_limit=60.0):
    """
    Optimize superellipse parameters (n_exp, aspect) to maximize min triangle area / hull area.
    """
    start_time = time.time()
    
    def neg_score(params):
        n_exp, aspect = params
        if n_exp < 1.01 or n_exp > 10.0 or aspect < 0.3 or aspect > 3.0:
            return 1.0
        return -_score_superellipse(n_exp, aspect)
    
    # Coarse grid search
    n_exp_vals = np.linspace(1.05, 8.0, 15)
    aspect_vals = np.linspace(0.5, 2.0, 12)
    
    best_score = 0.0
    best_n_exp = 2.0
    best_aspect = 1.0
    
    for ne in n_exp_vals:
        if time.time() > start_time + time_limit - 10:
            break
        for asp in aspect_vals:
            if time.time() > start_time + time_limit - 8:
                break
            s = _score_superellipse(ne, asp)
            if s > best_score:
                best_score = s
                best_n_exp = ne
                best_aspect = asp
    
    # Fine local optimization around best
    if time.time() < start_time + time_limit - 5:
        try:
            x0 = np.array([best_n_exp, best_aspect])
            res = minimize(
                neg_score, x0, method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < -best_score:
                best_n_exp, best_aspect = res.x
                best_score = -res.fun
        except Exception:
            pass
    
    pts = _superellipse_points(best_n_exp, best_aspect)
    return best_score, pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Superellipse optimization (new structural approach)
    if time.time() < deadline - 60:
        s1, pts1 = _optimize_superellipse(time_limit=min(50.0, deadline - 60 - time.time()))
        if s1 > best_score:
            best_score = s1
            best_points = pts1.copy()
    
    # Strategy 2: Regular 14-gon as baseline
    if time.time() < deadline - 40:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Strategy 3: Try a couple of specific superellipse configurations
    if time.time() < deadline - 30:
        for ne, asp in [(2.0, 1.0), (3.0, 1.2), (4.0, 1.0), (2.5, 0.9), (3.5, 1.1)]:
            if time.time() > deadline - 25:
                break
            s = _score_superellipse(ne, asp)
            if s > best_score:
                best_score = s
                best_points = _superellipse_points(ne, asp).copy()
    
    # Strategy 4: Final local polish
    if best_points is not None and time.time() < deadline - 15:
        time_for_polish = min(25.0, deadline - 15 - time.time())
        s, pts_opt = _local_optimize_free(best_points, time_limit=time_for_polish)
        if s > best_score:
            best_score = s
            best_points = pts_opt.copy()
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    final_score = _score(best_points)
    if final_score <= 0:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
