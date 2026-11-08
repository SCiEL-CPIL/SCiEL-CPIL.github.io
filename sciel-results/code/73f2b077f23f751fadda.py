"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools
from math import comb

N = 14

# EVOLVE-BLOCK-START
def _triangle_areas(points):
    """Compute all triangle areas for N points. Returns array of shape (C(N,3),)."""
    n = points.shape[0]
    areas = np.empty(comb(n, 3))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                areas[idx] = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                idx += 1
    return areas


def _convex_hull_area(points):
    """Compute convex hull area using Andrew's monotone chain."""
    pts = np.array(points, dtype=float)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    n = len(pts)
    if n < 3:
        return 0.0
    lower = []
    for p in pts:
        while len(lower) >= 2:
            x1, y1 = lower[-1]
            x2, y2 = p
            x0, y0 = lower[-2]
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2:
            x1, y1 = upper[-1]
            x2, y2 = p
            x0, y0 = upper[-2]
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if cross <= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return area


def _score(points):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _soft_min_neg_score(x, beta=50.0):
    """
    Smooth surrogate: minimize -soft_min(triangle_areas) / hull_area.
    We flatten x into 2*N coords. Returns scalar to minimize.
    """
    points = x.reshape(N, 2)
    areas = _triangle_areas(points)
    # Soft-min: min_i a_i ≈ -logsumexp(-a_i) / beta ... actually:
    # soft_min(a) = -(1/beta) * log(sum(exp(-beta * a)))
    # We want to maximize soft_min, so minimize -soft_min / hull_area
    # But hull_area is also variable. Let's work with log for stability.
    # Actually, let's just do: objective = -soft_min(areas) / hull_area
    # To make it differentiable and bounded, use:
    #   obj = -(1/beta) * log(sum(exp(-beta * areas))) / hull_area
    # But this can have issues. Alternative: use a penalty approach.
    
    # Simpler: use the soft-min as a smooth approximation to min, then divide by hull area.
    # For numerical stability, shift areas by their max.
    max_a = np.max(areas)
    shifted = areas - max_a  # all <= 0
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))  # this is log(sum(exp(-beta*(a - max_a))))
    # soft_min = max_a - (1/beta) * log_sum
    soft_min = max_a - (1.0 / beta) * log_sum
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10  # large penalty for degenerate hull
    
    return -soft_min / hull_area


def _gradient_soft_min_neg(x, beta=50.0):
    """
    Numerical gradient via central differences (N=14, 28 params, cheap enough).
    """
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _soft_min_neg_score(x, beta)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (_soft_min_neg_score(xp, beta) - _soft_min_neg_score(xm, beta)) / (2 * eps)
    return grad


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _jittered_polygon(n, radius=1.0, jitter=0.1, seed=42):
    """Regular polygon with small random jitter."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-jitter, jitter, n)
    radii = radius + rng.uniform(-jitter, jitter, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_points(n, seed=0):
    """Random points in unit square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, (n, 2))


def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    
    # Initial configurations to try
    initial_configs = []
    
    # 1. Regular polygon
    initial_configs.append(_regular_polygon(N, radius=1.0))
    
    # 2. Jittered polygons with different seeds
    for seed in range(10):
        initial_configs.append(_jittered_polygon(N, radius=1.0, jitter=0.05, seed=seed))
    
    # 3. Random points
    for seed in range(10):
        initial_configs.append(_random_points(N, seed=seed))
    
    # 4. Points on a circle with varying radii
    for seed in range(5):
        rng = np.random.RandomState(seed)
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = 0.5 + 0.5 * rng.uniform(0, 1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_configs.append(pts)
    
    # Optimize each initial config
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        x0 = init_pts.flatten()
        
        # Try L-BFGS-B with smooth objective
        try:
            result = minimize(
                _soft_min_neg_score,
                x0,
                method='L-BFGS-B',
                jac=_gradient_soft_min_neg,
                beta=30.0,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                opt_pts = result.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
        except Exception:
            pass
        
        # Also try with different beta values
        if time.time() > deadline:
            break
        
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline:
                break
            try:
                def obj_beta(x, b=beta):
                    return _soft_min_neg_score(x, b)
                
                def grad_beta(x, b=beta):
                    return _gradient_soft_min_neg(x, b)
                
                result = minimize(
                    obj_beta,
                    x0,
                    method='L-BFGS-B',
                    jac=grad_beta,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.fun < 0:
                    opt_pts = result.x.reshape(N, 2)
                    s = _score(opt_pts)
                    if s > best_score:
                        best_score = s
                        best_points = opt_pts.copy()
            except Exception:
                pass
    
    # If we haven't found anything good, use the best initial config score
    if best_points is None:
        for init_pts in initial_configs:
            s = _score(init_pts)
            if s > best_score:
                best_score = s
                best_points = init_pts.copy()
    
    # Final polish: try a few more local optimizations from the best point
    if best_points is not None and time.time() < deadline - 10:
        x_best = best_points.flatten()
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 5:
                break
            try:
                def obj_final(x, b=beta):
                    return _soft_min_neg_score(x, b)
                
                def grad_final(x, b=beta):
                    return _gradient_soft_min_neg(x, b)
                
                result = minimize(
                    obj_final,
                    x_best,
                    method='L-BFGS-B',
                    jac=grad_final,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < 0:
                    opt_pts = result.x.reshape(N, 2)
                    s = _score(opt_pts)
                    if s > best_score:
                        best_score = s
                        best_points = opt_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
