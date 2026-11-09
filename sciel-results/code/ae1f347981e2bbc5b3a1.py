"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 145.0  # seconds wall-clock limit; keep safety margin

# Precompute all triple indices
_ALL_TRIPLES = np.array(list(combinations(range(N), 3)))


def _all_triangle_areas(points):
    """Compute all triangle areas as a vector."""
    p1 = points[_ALL_TRIPLES[:, 0]]
    p2 = points[_ALL_TRIPLES[:, 1]]
    p3 = points[_ALL_TRIPLES[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    if len(points) < 3:
        return 0.0
    return float(np.min(_all_triangle_areas(points)))


def _hull_area(points):
    """Compute the convex hull area of the point set."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # In 2D, volume is area
    except Exception:
        return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return _min_triangle_area(points) / hull_area


def _negative_score(points):
    return -_score(points)


def _soft_min_triangle_area(points, beta=200.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp (soft-min).
    soft_min(a) = -(1/beta) * log(sum(exp(-beta * a_i)))
    As beta -> inf, this approaches the true min.
    """
    areas = _all_triangle_areas(points)
    areas = np.clip(areas, 1e-15, None)
    neg_beta_a = -beta * areas
    max_val = np.max(neg_beta_a)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_a - max_val)))
    soft_min = -log_sum / beta
    return float(soft_min)


def _soft_objective(x, beta=200.0):
    """
    Smooth objective: -soft_min_triangle_area / hull_area.
    Returns a smooth scalar to be minimized.
    """
    pts = x.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
    sm = _soft_min_triangle_area(pts, beta=beta)
    if sm < 1e-15:
        return 1e10
    return -sm / hull_area


def _exact_objective(x):
    """Exact nonsmooth objective for final polish."""
    pts = x.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
    min_area = _min_triangle_area(pts)
    return -min_area / hull_area


def _regular_polygon(n, perturb=0.0, rng_seed=42):
    """Create a regular n-gon, optionally perturbed."""
    rng = np.random.RandomState(rng_seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    if perturb > 0:
        pts += rng.randn(n, 2) * perturb
    return pts


def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf

    def _time_remaining():
        return DEADLINE - (time.time() - start_time)

    def _update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    def _regular_n_plus_interior(n_outer, n_interior, perturb=0.0, rng_seed=42):
        """n_outer points on a regular polygon + n_interior points near center."""
        outer = _regular_polygon(n_outer, perturb=perturb, rng_seed=rng_seed)
        rng = np.random.RandomState(rng_seed)
        if n_interior == 1:
            interior = np.array([[0.0, 0.0]])
        elif n_interior == 2:
            interior = np.array([[0.03, 0.0], [-0.03, 0.0]])
        elif n_interior == 3:
            interior = np.array([[0.03, 0.0], [-0.015, 0.026], [-0.015, -0.026]])
        else:
            interior = rng.randn(n_interior, 2) * 0.02
        return np.vstack([outer, interior])

    # Build diverse initial configurations
    configs = []

    # 12-gon + center (primary topology)
    configs.append(_regular_n_plus_interior(12, 1, perturb=0.0, rng_seed=42))

    # Offset centers for 12-gon
    for dx in [-0.1, 0.1, -0.05, 0.05, 0.0]:
        for dy in [-0.1, 0.1, -0.05, 0.05, 0.0]:
            if dx == 0 and dy == 0:
                continue
            rng = np.random.RandomState(42 + int(abs(dx*100) + abs(dy*100)))
            angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
            center = np.array([[dx, dy]])
            configs.append(np.vstack([pts, center]))

    # Perturbed 12-gons
    for perturb in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            configs.append(_regular_n_plus_interior(12, 1, perturb=perturb, rng_seed=seed))

    # Random near 12-gon with varying radii
    for seed in range(25):
        rng2 = np.random.RandomState(seed)
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng2.randn(12) * 0.02
        radii = 1.0 + rng2.randn(12) * 0.06
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        center = rng2.randn(2) * 0.06
        configs.append(np.vstack([pts, np.array([center])]))

    # 11-gon + 2 interior
    for seed in range(8):
        configs.append(_regular_n_plus_interior(11, 2, perturb=0.01, rng_seed=seed))

    # 10-gon + 3 interior
    for seed in range(8):
        configs.append(_regular_n_plus_interior(10, 3, perturb=0.01, rng_seed=seed))

    # Regular 13-gon (all on hull)
    angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    configs.append(np.column_stack([np.cos(angles13), np.sin(angles13)]))

    # Perturbed 13-gon
    for seed in range(5):
        rng13 = np.random.RandomState(100 + seed)
        angles13p = np.linspace(0, 2 * np.pi, 13, endpoint=False) + rng13.randn(13) * 0.02
        radii13 = 1.0 + rng13.randn(13) * 0.05
        pts13 = np.column_stack([radii13 * np.cos(angles13p), radii13 * np.sin(angles13p)])
        configs.append(pts13)

    # Evaluate all configs with exact score
    for cfg in configs:
        if _time_remaining() < 5.0:
            break
        _update_best(cfg)

    # Sort configs by score for prioritized optimization
    scored_configs = []
    for cfg in configs:
        s = _score(cfg)
        scored_configs.append((s, cfg))
    scored_configs.sort(key=lambda x: -x[0])

    # Phase 1: L-BFGS-B with soft-min objective, progressive beta annealing
    betas = [30.0, 50.0, 80.0, 120.0, 200.0, 350.0, 500.0, 800.0]
    
    for beta in betas:
        if _time_remaining() < 15.0:
            break
        # Number of starts to try at this beta level
        n_starts = min(6, len(scored_configs))
        for s, cfg in scored_configs[:n_starts]:
            if _time_remaining() < 5.0:
                break
            x0 = cfg.flatten()
            try:
                result = minimize(
                    _soft_objective, x0, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                opt_pts = result.x.reshape(-1, 2)
                _update_best(opt_pts)
            except Exception:
                pass

    # Phase 2: Nelder-Mead polish with exact objective from best result
    if _time_remaining() > 12.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    # Phase 3: Powell method (good for nonsmooth, coordinate-based)
    if _time_remaining() > 10.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-12, 'xtol': 1e-10}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    # Phase 4: Additional Nelder-Mead from other good starts
    if _time_remaining() > 8.0:
        for s, cfg in scored_configs[:4]:
            if _time_remaining() < 4.0:
                break
            x0 = cfg.flatten()
            try:
                result = minimize(
                    _exact_objective, x0, method='Nelder-Mead',
                    options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                opt_pts = result.x.reshape(-1, 2)
                _update_best(opt_pts)
            except Exception:
                pass

    # Phase 5: Final high-precision L-BFGS-B with very sharp soft-min
    if _time_remaining() > 6.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _soft_objective, x0, args=(1500.0,),
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    # Phase 6: Final exact Nelder-Mead with tight tolerances
    if _time_remaining() > 3.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Nelder-Mead',
                options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    # Phase 7: One more Powell pass if time allows
    if _time_remaining() > 2.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Powell',
                options={'maxiter': 2000, 'ftol': 1e-13, 'xtol': 1e-11}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    return best_points
