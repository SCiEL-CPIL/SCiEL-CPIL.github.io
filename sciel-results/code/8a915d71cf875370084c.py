"""Heilbronn convex regions n=14 — two concentric heptagons with analytical radius + 1D rotation search."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar, minimize

N = 14

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area."""
    n = len(points)
    if n < 3:
        return 0.0
    # Use cross product: area = 0.5 * |cross(b-a, c-a)|
    # For all triples i<j<k
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area using scipy.spatial.ConvexHull."""
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
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _two_heptagons(r, theta):
    """
    Generate 14 points: 7 on outer unit circle, 7 on inner circle of radius r,
    with inner ring rotated by theta relative to outer ring.
    """
    n_out = 7
    n_in = 7
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _score_two_heptagons(r, theta):
    """Score for two heptagons configuration."""
    pts = _two_heptagons(r, theta)
    return _score(pts)

def _neg_score_two_heptagons(params):
    """Negative score for optimization."""
    r, theta = params
    if r <= 0 or r > 1:
        return 1.0
    return -_score_two_heptagons(r, theta)

def _optimize_1d_rotation(r, time_limit=5.0):
    """Optimize rotation angle for a fixed inner radius."""
    start_time = time.time()
    
    def neg_theta(theta):
        return -_score_two_heptagons(r, theta)
    
    # Search over theta in [0, pi/7) since the heptagon has 7-fold symmetry
    # Actually, we need to consider the relative rotation, which has period 2*pi/7
    # But let's search over [0, pi] to be safe, then use symmetry
    
    best_score = 0.0
    best_theta = 0.0
    
    # Coarse scan
    n_scan = 50
    thetas = np.linspace(0, np.pi / 7, n_scan, endpoint=False)
    for th in thetas:
        s = _score_two_heptagons(r, th)
        if s > best_score:
            best_score = s
            best_theta = th
    
    # Fine local optimization around best
    if time.time() < start_time + time_limit:
        try:
            res = minimize_scalar(
                neg_theta,
                bounds=(max(0, best_theta - np.pi / 28), min(np.pi / 7, best_theta + np.pi / 28)),
                method='bounded',
                options={'xatol': 1e-8}
            )
            if time.time() < start_time + time_limit:
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_theta = res.x
        except Exception:
            pass
    
    return best_score, best_theta

def _optimize_radius_1d(theta, time_limit=5.0):
    """Optimize inner radius for a fixed rotation angle."""
    start_time = time.time()
    
    def neg_r(r):
        return -_score_two_heptagons(r, theta)
    
    best_score = 0.0
    best_r = 0.5
    
    # Coarse scan
    n_scan = 30
    radii = np.linspace(0.1, 0.95, n_scan)
    for rr in radii:
        s = _score_two_heptagons(rr, theta)
        if s > best_score:
            best_score = s
            best_r = rr
    
    # Fine local optimization
    if time.time() < start_time + time_limit:
        try:
            res = minimize_scalar(
                neg_r,
                bounds=(max(0.05, best_r - 0.1), min(0.98, best_r + 0.1)),
                method='bounded',
                options={'xatol': 1e-8}
            )
            if time.time() < start_time + time_limit:
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_r = res.x
        except Exception:
            pass
    
    return best_score, best_r

def _local_optimize_free(points, time_limit=10.0):
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
            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if time.time() < start_time + time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
    
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Two concentric heptagons with joint (r, theta) optimization
    # Start with a grid of (r, theta) values
    r_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    theta_values = [0.0, np.pi / 14, np.pi / 7, 2 * np.pi / 14, 3 * np.pi / 14, np.pi / 4]
    
    # Coarse grid search
    for r in r_values:
        if time.time() > deadline - 30:
            break
        for theta in theta_values:
            if time.time() > deadline - 25:
                break
            s = _score_two_heptagons(r, theta)
            if s > best_score:
                best_score = s
                best_points = _two_heptagons(r, theta).copy()
    
    # Refine: for each promising (r, theta), do 1D optimization
    if best_points is not None and time.time() < deadline - 40:
        # Extract current r and theta from best_points
        # The outer ring has 7 points on unit circle, inner ring has 7 points on circle of radius r
        # We can estimate r from the inner points
        pts = best_points
        # Inner points are those with norm < 0.99
        norms = np.linalg.norm(pts, axis=1)
        inner_mask = norms < 0.99
        if np.sum(inner_mask) == 7:
            r_est = np.mean(norms[inner_mask])
            # Estimate theta from inner points
            inner_pts = pts[inner_mask]
            angles_in = np.arctan2(inner_pts[:, 1], inner_pts[:, 0])
            # The outer points define the reference
            outer_mask = ~inner_mask
            outer_pts = pts[outer_mask]
            angles_out = np.arctan2(outer_pts[:, 1], outer_pts[:, 0])
            # theta is the offset
            theta_est = (np.mean(angles_in) - np.mean(angles_out)) % (2 * np.pi)
        else:
            r_est = 0.5
            theta_est = 0.0
        
        # Optimize theta for current r
        s_theta, theta_opt = _optimize_1d_rotation(r_est, time_limit=8.0)
        if s_theta > best_score:
            best_score = s_theta
            best_points = _two_heptagons(r_est, theta_opt).copy()
        
        # Optimize r for current theta
        s_r, r_opt = _optimize_radius_1d(theta_opt, time_limit=8.0)
        if s_r > best_score:
            best_score = s_r
            best_points = _two_heptagons(r_opt, theta_opt).copy()
        
        # Joint refinement: alternate r and theta optimization
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            # Get current r, theta
            pts = best_points
            norms = np.linalg.norm(pts, axis=1)
            inner_mask = norms < 0.99
            if np.sum(inner_mask) == 7:
                r_cur = np.mean(norms[inner_mask])
                inner_pts = pts[inner_mask]
                outer_pts = pts[~inner_mask]
                angles_in = np.arctan2(inner_pts[:, 1], inner_pts[:, 0])
                angles_out = np.arctan2(outer_pts[:, 1], outer_pts[:, 0])
                theta_cur = (np.mean(angles_in) - np.mean(angles_out)) % (2 * np.pi)
            else:
                break
            
            # Optimize theta
            s1, theta_new = _optimize_1d_rotation(r_cur, time_limit=5.0)
            if s1 > best_score:
                best_score = s1
                best_points = _two_heptagons(r_cur, theta_new).copy()
            
            # Optimize r
            s2, r_new = _optimize_radius_1d(theta_new, time_limit=5.0)
            if s2 > best_score:
                best_score = s2
                best_points = _two_heptagons(r_new, theta_new).copy()
    
    # Strategy 2: Try some other symmetric configurations
    if time.time() < deadline - 30:
        # Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        
        # Two rings with different counts: 8 outer, 6 inner
        for r in [0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline - 20:
                break
            angles_out = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
            pts = np.vstack([
                np.column_stack((np.cos(angles_out), np.sin(angles_out))),
                np.column_stack((r * np.cos(angles_in), r * np.sin(angles_in)))
            ])
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        
        # 6 outer, 8 inner
        for r in [0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline - 15:
                break
            angles_out = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            angles_in = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
            pts = np.vstack([
                np.column_stack((np.cos(angles_out), np.sin(angles_out))),
                np.column_stack((r * np.cos(angles_in), r * np.sin(angles_in)))
            ])
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
    
    # Strategy 3: Final local polish
    if best_points is not None and time.time() < deadline - 10:
        s, pts_opt = _local_optimize_free(best_points, time_limit=15.0)
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
