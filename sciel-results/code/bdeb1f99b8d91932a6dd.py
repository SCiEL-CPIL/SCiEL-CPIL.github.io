"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _regular_polygon_angles(n=N):
    return np.linspace(0, 2 * np.pi, n, endpoint=False)

def _elliptical_pts(A, B, C):
    """
    Generate 13 points by applying elliptical distortion to a regular 13-gon.
    Matrix M = [[A, B], [B, C]] applied to unit circle points.
    """
    angles = _regular_polygon_angles(N)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    x = A * cos_a + B * sin_a
    y = B * cos_a + C * sin_a
    return np.column_stack([x, y])

def _elliptical_objective(theta, beta=50.0):
    """
    Negative soft-min of (min triangle area / hull area) for elliptical params.
    theta = [A, B, C], with constraint A*C > B^2 (positive definite).
    We parameterize with A>0, C>0, and B in (-1,1) scaled to ensure PD.
    """
    A, B_raw, C = theta
    # Ensure positive definite: use B = B_raw * 0.9 * sqrt(A*C)
    if A <= 0 or C <= 0:
        return 1e6
    B = B_raw * 0.9 * np.sqrt(A * C)
    pts = _elliptical_pts(A, B, C)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _elliptical_objective_grad(theta, beta=50.0):
    """Gradient of elliptical objective using numerical differentiation."""
    eps = 1e-7
    grad = np.zeros(3)
    f0 = _elliptical_objective(theta, beta)
    for i in range(3):
        theta_p = theta.copy()
        theta_p[i] += eps
        f_p = _elliptical_objective(theta_p, beta)
        grad[i] = (f_p - f0) / eps
    return grad

def _adaptive_soft_min_optimize_elliptical(theta0, time_budget=10.0):
    """
    Optimize elliptical parameters with adaptive beta (increasing sharpness).
    Returns best points found.
    """
    start = time.time()
    best_pts = _elliptical_pts(*theta0)
    best_score = _score(best_pts)
    
    for beta in [10.0, 30.0, 80.0, 200.0, 500.0]:
        if time.time() - start > time_budget:
            break
        try:
            result = minimize(
                lambda t: _elliptical_objective(t, beta),
                theta0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            theta_opt = result.x
            A, B_raw, C = theta_opt
            if A > 0 and C > 0:
                B = B_raw * 0.9 * np.sqrt(A * C)
                pts = _elliptical_pts(A, B, C)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts.copy()
                    theta0 = theta_opt.copy()
        except Exception:
            continue
    
    return best_pts, best_score

def _soft_min_objective_cart(x, beta=50.0):
    """Negative soft-min for Cartesian coordinates."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _exact_objective(x):
    """Negative exact score for minimization (non-smooth)."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _nelder_mead_polish(pts, max_iter=300, tol=1e-12):
    """Polish using Nelder-Mead on the exact objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            _exact_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def _concentric_ring_init(seed=0, n_inner=5, n_outer=8):
    """Generate points on two concentric rings."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_in = 0.4 + rng.uniform(-0.1, 0.1)
    pts[:n_inner, 0] = r_in * np.cos(angles_in)
    pts[:n_inner, 1] = r_in * np.sin(angles_in)
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_out = 1.0 + rng.uniform(-0.1, 0.1)
    pts[n_inner:, 0] = r_out * np.cos(angles_out)
    pts[n_inner:, 1] = r_out * np.sin(angles_out)
    return pts

def _regular_polygon_init(seed=0):
    """Regular polygon with small perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
    r = 1.0 + rng.normal(0, 0.03, N)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    best_score = 0.0
    best_pts = _regular_polygon_init(0)
    sc = _score(best_pts)
    if sc > best_score:
        best_score = sc
    
    # Phase 1: Elliptical parameterization optimization (3 params, smooth landscape)
    # Try multiple initial elliptical shapes
    elliptical_inits = [
        [1.0, 0.0, 1.0],      # circle
        [1.2, 0.0, 0.8],      # horizontal ellipse
        [0.8, 0.0, 1.2],      # vertical ellipse
        [1.0, 0.3, 1.0],      # tilted
        [1.0, -0.3, 1.0],     # tilted other way
        [1.5, 0.0, 0.7],      # stretched
        [0.7, 0.0, 1.5],      # stretched other
        [1.1, 0.2, 0.9],      # mild tilt
        [1.0, 0.1, 1.1],      # near circle
        [1.3, -0.2, 0.85],    # asymmetric
    ]
    
    for i, theta0 in enumerate(elliptical_inits):
        if time.time() > deadline - 40:
            break
        try:
            pts, sc = _adaptive_soft_min_optimize_elliptical(theta0, time_budget=8.0)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 2: Cartesian multi-start with soft-min L-BFGS-B
    init_strategies = [
        _regular_polygon_init,
        _concentric_ring_init,
        lambda s: _elliptical_pts(1.0, 0.0, 1.0) + np.random.RandomState(s).normal(0, 0.1, (N, 2)),
    ]
    
    n_starts = 15
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        strategy_idx = i % len(init_strategies)
        seed = i + 50
        try:
            pts0 = init_strategies[strategy_idx](seed)
            x0 = pts0.flatten()
            
            # Adaptive beta for Cartesian
            for beta in [20.0, 60.0, 150.0]:
                if time.time() > deadline - 25:
                    break
                result = minimize(
                    lambda x: _soft_min_objective_cart(x, beta),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                pts = result.x.reshape(-1, 2)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts.copy()
                x0 = pts.flatten()
        except Exception:
            continue
    
    # Phase 3: High-precision Nelder-Mead polish on best solution
    if time.time() < deadline - 15:
        polished = _nelder_mead_polish(best_pts, max_iter=500, tol=1e-13)
        sc = _score(polished)
        if sc > best_score:
            best_score = sc
            best_pts = polished.copy()
    
    # Phase 4: If time permits, a few more diverse starts with full polish
    for seed in range(5):
        if time.time() > deadline - 8:
            break
        try:
            pts0 = _concentric_ring_init(seed + 200)
            x0 = pts0.flatten()
            result = minimize(
                lambda x: _soft_min_objective_cart(x, 100.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            pts = result.x.reshape(-1, 2)
            pts = _nelder_mead_polish(pts, max_iter=300, tol=1e-12)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
        except Exception:
            continue
    
    return best_pts
# EVOLVE-BLOCK-END
