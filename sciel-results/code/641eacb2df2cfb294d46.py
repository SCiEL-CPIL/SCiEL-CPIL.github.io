"""Heilbronn convex regions n=14: multi-start optimization of min-triangle/hull-area.

The score is (min triangle area) / (convex hull area), scale-invariant.
For points in convex position, the minimum-area triangle is typically among
three consecutive points when sorted by angle, but we verify all triples.
We use angle-based parameterization for efficiency and multiple optimization
methods with diverse starting points to find the optimal configuration.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14


def _all_triangle_areas(points):
    """Compute all C(n,3) triangle areas."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) -
                              (y[j] - y[i]) * (x[k] - x[i]))
                areas.append(a)
    return np.array(areas)


def _min_triangle_area(points):
    """Exact minimum triangle area over all C(n,3) triples."""
    return np.min(_all_triangle_areas(points))


def _hull_area(points):
    try:
        return ConvexHull(points).volume
    except Exception:
        return 0.0


def _score(points):
    """min triangle area / convex hull area."""
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 0.0
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0.0:
        return 0.0
    return ma / ha


def _regular_polygon(k, r=1.0, phase=0.0):
    a = phase + 2 * np.pi * np.arange(k) / k
    return r * np.column_stack([np.cos(a), np.sin(a)])


def _pts_from_angles(theta, r=1.0):
    """Points on unit circle from angles (not necessarily sorted)."""
    th = np.asarray(theta, dtype=float)
    return r * np.column_stack([np.cos(th), np.sin(th)])


def _consecutive_min_triangle_area(theta):
    """Fast O(n) min triangle area assuming points on circle, sorted by angle.
    Only checks consecutive triples which is valid for convex position."""
    th = np.sort(theta)
    n = len(th)
    min_area = float('inf')
    for i in range(n):
        i1, i2, i3 = i % n, (i+1) % n, (i+2) % n
        d1 = th[i2] - th[i1]
        d2 = th[i3] - th[i2]
        # Area of triangle with vertices at angles separated by d1, d2 on unit circle
        # = 0.5 * |sin(d1) * sin(d2) * sin(2*pi - d1 - d2)| but simpler to compute directly
        x1, y1 = np.cos(th[i1]), np.sin(th[i1])
        x2, y2 = np.cos(th[i2]), np.sin(th[i2])
        x3, y3 = np.cos(th[i3]), np.sin(th[i3])
        a = 0.5 * abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
        if a < min_area:
            min_area = a
    return min_area


def _fast_ratio_from_angles(theta):
    """Fast ratio using consecutive triples only (valid for convex position on circle)."""
    th = np.sort(theta)
    min_area = _consecutive_min_triangle_area(th)
    # Hull area for points on unit circle in convex position = area of polygon
    # which can be computed as sum of cross products
    x = np.cos(th)
    y = np.sin(th)
    # Shoelace formula
    ha = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if ha < 1e-12:
        return 0.0
    return min_area / ha


def _objective_from_angles(theta_flat):
    """Objective for minimization: minimize -ratio (maximize ratio).
    Uses fast consecutive-triple approximation for speed."""
    th = np.asarray(theta_flat, dtype=float)
    if th.shape != (N,) or np.any(~np.isfinite(th)):
        return 1e10
    ratio = _fast_ratio_from_angles(th)
    if ratio < 1e-12:
        return 1e10
    return -ratio


def _objective_exact_from_flat(flat):
    """Objective using exact min over all triples (slower but accurate)."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    s = _score(pts)
    if s < 1e-12:
        return 1e10
    return -s


def _nm_optimize(x0, deadline, obj, maxiter=5000):
    """Nelder-Mead with time guard."""
    if time.time() > deadline:
        return None
    try:
        result = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-12,
                                   'fatol': 1e-14, 'adaptive': True,
                                   'maxfev': maxiter * 20})
        if time.time() > deadline:
            return None
        return result
    except Exception:
        return None


def _powell_optimize(x0, deadline, obj, maxiter=1000):
    """Powell with time guard."""
    if time.time() > deadline:
        return None
    try:
        result = minimize(obj, x0, method='Powell',
                          options={'maxiter': maxiter, 'xtol': 1e-14,
                                   'ftol': 1e-16})
        if time.time() > deadline:
            return None
        return result
    except Exception:
        return None


def _lbfgs_optimize(x0, deadline, obj, maxiter=2000):
    """L-BFGS-B with time guard."""
    if time.time() > deadline:
        return None
    try:
        result = minimize(obj, x0, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-14,
                                   'gtol': 1e-12, 'maxfun': maxiter * 50})
        if time.time() > deadline:
            return None
        return result
    except Exception:
        return None


def _consider(pts, best):
    """Update best if pts gives a better score."""
    if pts is None or pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
        return
    s = _score(pts)
    if s > best['score']:
        best['score'] = s
        best['pts'] = pts.copy()


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


def _generate_starts(deadline):
    """Generate diverse starting configurations as angle arrays."""
    starts = []
    t0 = time.time()
    
    # 1. Regular 14-gon (baseline)
    base_angles = 2 * np.pi * np.arange(N) / N
    starts.append(base_angles.copy())
    
    # 2. Regular 14-gon with various phases
    for phase in [np.pi/14, np.pi/7, 3*np.pi/14, np.pi/3, np.pi/2]:
        if time.time() > deadline - 40:
            break
        starts.append((base_angles + phase) % (2*np.pi))
    
    # 3. Perturbed regular polygons (random small perturbations)
    rng = np.random.default_rng(42)
    for trial in range(30):
        if time.time() > deadline - 40:
            break
        dtheta = rng.normal(0, 0.05 + 0.02 * trial, N)
        angles = base_angles + dtheta
        starts.append(angles)
    
    # 4. Random angles (sorted)
    for trial in range(20):
        if time.time() > deadline - 40:
            break
        rng = np.random.default_rng(100 + trial)
        angles = np.sort(rng.uniform(0, 2*np.pi, N))
        starts.append(angles)
    
    # 5. Clustered angles (some points close together)
    for trial in range(10):
        if time.time() > deadline - 40:
            break
        rng = np.random.default_rng(200 + trial)
        # Create clusters
        cluster_centers = rng.uniform(0, 2*np.pi, 3)
        angles = []
        for i in range(N):
            c = cluster_centers[i % 3]
            angles.append(c + rng.normal(0, 0.1))
        angles = np.sort(angles)
        starts.append(angles)
    
    # 6. Alternating large/small gaps
    if time.time() < deadline - 40:
        for ratio in [1.2, 1.5, 2.0]:
            gaps = np.ones(N)
            gaps[::2] *= ratio
            gaps /= np.sum(gaps) * 2 * np.pi
            angles = np.cumsum(gaps)
            angles[0] = 0
            starts.append(angles)
    
    return starts


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin
    best = {'score': 0.0, 'pts': None}
    
    # Evaluate regular polygon first
    reg_pts = _regular_polygon(N)
    _consider(reg_pts, best)
    
    # Generate diverse starts
    starts = _generate_starts(deadline)
    
    # Optimize from each start using multiple methods
    for idx, theta0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        # Method 1: Nelder-Mead on angles (fast objective)
        if time.time() < deadline - 15:
            r = _nm_optimize(theta0.copy(), deadline, _objective_from_angles, maxiter=3000)
            if r is not None and np.isfinite(r.fun):
                cand_pts = _pts_from_angles(r.x)
                _consider(cand_pts, best)
        
        # Method 2: Powell on angles
        if time.time() < deadline - 15:
            r = _powell_optimize(theta0.copy(), deadline, _objective_from_angles, maxiter=500)
            if r is not None and np.isfinite(r.fun):
                cand_pts = _pts_from_angles(r.x)
                _consider(cand_pts, best)
        
        # Method 3: L-BFGS on angles
        if time.time() < deadline - 15:
            r = _lbfgs_optimize(theta0.copy(), deadline, _objective_from_angles, maxiter=1000)
            if r is not None and np.isfinite(r.fun):
                cand_pts = _pts_from_angles(r.x)
                _consider(cand_pts, best)
        
        # Method 4: Direct 2D optimization from best angle result
        if time.time() < deadline - 10:
            # Take the best angle result so far and optimize in 2D
            if best['pts'] is not None:
                flat0 = best['pts'].flatten()
                r = _nm_optimize(flat0.copy(), deadline, _objective_exact_from_flat, maxiter=2000)
                if r is not None and np.isfinite(r.fun):
                    _consider(r.x.reshape(N, 2), best)
    
    # Final polish: if we have a good candidate, do one more NM pass in 2D
    if best['pts'] is not None and time.time() < deadline - 5:
        flat0 = best['pts'].flatten()
        r = _nm_optimize(flat0.copy(), deadline, _objective_exact_from_flat, maxiter=5000)
        if r is not None and np.isfinite(r.fun):
            _consider(r.x.reshape(N, 2), best)
    
    # Fallback: ensure we return something valid
    if best['pts'] is None or best['score'] < 1e-6:
        best['pts'] = _regular_polygon(N)
        best['score'] = _score(best['pts'])
    
    return _normalize_unit_area(best['pts'])
