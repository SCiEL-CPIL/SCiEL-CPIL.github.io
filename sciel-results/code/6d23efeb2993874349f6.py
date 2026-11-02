"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
import time
from itertools import combinations

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples (vectorized)."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                                  (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1]))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute convex hull area using Andrew's monotone chain."""
    pts = np.array(pts, dtype=float)
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _elliptic_boundary(a, b, n_samples=400):
    """Generate points on an ellipse with semi-axes a, b."""
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    return np.column_stack([a * np.cos(theta), b * np.sin(theta)])


def _parametric_boundary(params, n_samples=400):
    """
    Generate a convex boundary from parameters.
    params: [a, b, h2c, h2s, h3c, h3s, h4c, h4s, ...]
    a, b: ellipse semi-axes
    Remaining pairs: harmonic (cos, sin) coefficients for r(theta) modulation
    """
    a, b = params[0], params[1]
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    
    # Base ellipse
    x_base = a * np.cos(theta)
    y_base = b * np.sin(theta)
    
    # Add harmonic perturbations to radius
    n_harmonics = (len(params) - 2) // 2
    for k in range(n_harmonics):
        harmonic_num = k + 2  # start from 2nd harmonic
        ck = params[2 + 2 * k]
        sk = params[2 + 2 * k + 1]
        # Modulate the ellipse radius
        r_mod = ck * np.cos(harmonic_num * theta) + sk * np.sin(harmonic_num * theta)
        # Scale factor applied to base ellipse
        scale = 1.0 + 0.15 * r_mod  # small modulation
        x_base *= scale
        y_base *= scale
    
    return np.column_stack([x_base, y_base])


def _place_by_angle(boundary_pts, n_points, angles):
    """
    Place n_points on boundary at given angles (radians, 0 to 2*pi).
    Uses interpolation along the boundary.
    """
    n_b = len(boundary_pts)
    # Compute boundary angles
    bx, by = boundary_pts[:, 0], boundary_pts[:, 1]
    b_angles = np.arctan2(by, bx)
    b_angles = np.where(b_angles < 0, b_angles + 2 * np.pi, b_angles)
    
    # Sort boundary by angle
    order = np.argsort(b_angles)
    b_sorted = boundary_pts[order]
    b_angles_sorted = b_angles[order]
    
    points = np.zeros((n_points, 2))
    for i in range(n_points):
        ang = angles[i] % (2 * np.pi)
        # Find position by angle interpolation
        # Handle wrap-around
        diff = b_angles_sorted - ang
        diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
        diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)
        idx = np.argmin(np.abs(diff))
        # Interpolate between idx and idx+1
        idx1 = idx % n_b
        idx2 = (idx + 1) % n_b
        a1 = b_angles_sorted[idx1]
        a2 = b_angles_sorted[idx2]
        # Handle wrap
        if a2 < a1:
            a2 += 2 * np.pi
        if ang < a1:
            ang += 2 * np.pi
        t = (ang - a1) / (a2 - a1 + 1e-15)
        t = np.clip(t, 0, 1)
        points[i] = (1 - t) * b_sorted[idx1] + t * b_sorted[idx2]
    
    return points


def _optimize_points_nm(initial_pts, max_iter=200, timeout=None):
    """
    Optimize point positions using Nelder-Mead to maximize min triangle area / hull area.
    """
    from scipy.optimize import minimize
    
    def objective(x):
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    x0 = initial_pts.flatten()
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
    
    best_pts = result.x.reshape(N, 2)
    best_score = -result.fun
    return best_pts, best_score


def _optimize_ellipse_aspect(initial_pts, max_iter=100):
    """
    Optimize by scaling x and y coordinates independently (ellipse aspect ratio).
    """
    best_pts = initial_pts.copy()
    best_score = _score(best_pts)
    
    for _ in range(max_iter):
        improved = False
        for sx in [0.9, 0.95, 1.0, 1.05, 1.1]:
            for sy in [0.9, 0.95, 1.0, 1.05, 1.1]:
                scaled = best_pts.copy()
                scaled[:, 0] *= sx
                scaled[:, 1] *= sy
                s = _score(scaled)
                if s > best_score + 1e-10:
                    best_score = s
                    best_pts = scaled
                    improved = True
        if not improved:
            break
    
    return best_pts, best_score


def construct():
    """
    Construct 14 points maximizing min triangle area / convex hull area.
    Structural approach: parametric convex boundary + direct Nelder-Mead optimization
    of point positions, with ellipse aspect ratio as a key design parameter.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_points = None
    best_score = 0.0
    rng = np.random.RandomState(12345)
    
    # Strategy 1: Regular polygon baseline
    theta_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(theta_reg), np.sin(theta_reg)])
    reg_score = _score(reg_pts)
    best_points = reg_pts.copy()
    best_score = reg_score
    
    # Strategy 2: Ellipse with optimized aspect ratio
    for aspect in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        if time.time() > deadline - 20:
            break
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack([aspect * np.cos(theta), np.sin(theta)])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Strategy 3: Parametric boundary with harmonics + angle-based placement
    n_samples = 400
    for n_harm in [0, 1, 2]:
        for trial in range(15):
            if time.time() > deadline - 15:
                break
            
            # Generate parameters
            a = 1.0 + rng.uniform(-0.2, 0.2)
            b = 1.0 + rng.uniform(-0.2, 0.2)
            params = np.array([a, b])
            for k in range(n_harm):
                scale = 0.3 / (k + 2)
                params = np.append(params, [rng.uniform(-scale, scale),
                                             rng.uniform(-scale, scale)])
            
            boundary = _parametric_boundary(params, n_samples)
            
            # Check convexity
            hull_area = _convex_hull_area(boundary)
            if hull_area < 0.9 * _convex_hull_area(np.column_stack(
                    [np.cos(np.linspace(0, 2*np.pi, n_samples, endpoint=False)),
                     np.sin(np.linspace(0, 2*np.pi, n_samples, endpoint=False))])):
                continue
            
            # Place points at equal angles
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            pts = _place_by_angle(boundary, N, angles)
            s = _score(pts)
            
            if s > best_score:
                best_score = s
                best_points = pts.copy()
            
            # Optimize aspect ratio of this placement
            if time.time() < deadline - 10:
                opt_pts, opt_score = _optimize_ellipse_aspect(pts, max_iter=50)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
    
    # Strategy 4: Nelder-Mead optimization from best point set
    if best_points is not None and time.time() < deadline - 30:
        nm_pts, nm_score = _optimize_points_nm(best_points, max_iter=150)
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_pts.copy()
    
    # Strategy 5: Multiple Nelder-Mead restarts from perturbed baselines
    if time.time() < deadline - 10:
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
            perturb = rng.normal(0, 0.08, (N, 2))
            pts = np.column_stack([np.cos(theta), np.sin(theta)]) + perturb
            nm_pts, nm_score = _optimize_points_nm(pts, max_iter=100)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_pts.copy()
    
    # Strategy 6: Final fine-tuning with small perturbations
    if best_points is not None and time.time() < deadline - 3:
        for trial in range(100):
            if time.time() > deadline - 1:
                break
            perturbed = best_points + rng.normal(0, 0.02, (N, 2))
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_points = perturbed.copy()
    
    return best_points

# EVOLVE-BLOCK-END
