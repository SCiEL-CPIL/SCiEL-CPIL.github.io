"""Fresh design: symmetric parameterization + smooth min-area surrogate + L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize

N = 13
_DEADLINE = time.time() + 120.0  # explicit wall-clock guard (safety margin vs 150s kill)


def _min_tri_area(pts):
    """Exact minimum triangle area (absolute) over all triples."""
    n = len(pts)
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            for k in range(j + 1, n):
                w = pts[k] - pts[i]
                a = abs(v[0] * w[1] - v[1] * w[0]) * 0.5
                if a < best:
                    best = a
    return best


def _hull_area(pts):
    """Area of the convex hull of pts."""
    from scipy.spatial import ConvexHull
    try:
        return ConvexHull(pts).volume  # 2-D ConvexHull stores area in .volume
    except Exception:
        return 1.0


def _soft_min_area(pts, beta):
    """Smooth surrogate: soft-min of triangle areas via log-sum-exp (lower bound of true min)."""
    n = len(pts)
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            for k in range(j + 1, n):
                w = pts[k] - pts[i]
                areas[i, j, k] = abs(v[0] * w[1] - v[1] * w[0]) * 0.5
    m = areas.min()
    s = np.sum(np.exp(beta * (areas - m)))
    return m - np.log(s) / beta


def _score_from_params(params, beta):
    """Map params -> points, return (soft_min_area, hull_area)."""
    pts = _params_to_points(params)
    soft = _soft_min_area(pts, beta)
    hull = _hull_area(pts)
    return soft, hull


def _params_to_points(params):
    """
    Symmetric parameterization:
      - 7 outer vertices of a heptagon-like ring with radius r0 and angular offsets
      - 5 interior points on 5 symmetry axes with radial distances
      - 1 center-ish point
    Total 13. We use a regular-7 outer ring plus 5 interior on symmetry lines + 1 center.
    """
    # params: [r0, ang_off, *5 interior radii, center_r, center_ang]
    r0 = params[0]
    ang_off = params[1]
    # 7 outer vertices
    k = np.arange(7)
    th = 2 * np.pi * k / 7.0 + ang_off
    outer = np.column_stack([r0 * np.cos(th), r0 * np.sin(th)])
    # 5 interior points on 5 symmetry axes (angles = outer vertex angles for 5 of them)
    int_r = params[2:7]
    int_th = 2 * np.pi * np.arange(5) / 7.0 + ang_off
    interior = np.column_stack([int_r * np.cos(int_th), int_r * np.sin(int_th)])
    # 1 center point
    cr = params[7]
    ca = params[8]
    center = np.array([[cr * np.cos(ca), cr * np.sin(ca)]])
    pts = np.vstack([outer, interior, center])
    return pts


def _objective(params, beta):
    soft, hull = _score_from_params(params, beta)
    if hull <= 1e-9:
        return 1e12
    return -(soft / hull)  # maximize soft_min/hull


def _neg_obj(params, beta):
    return -_objective(params, beta)


def construct():
    global _DEADLINE
    t0 = time.time()

    # Initial guess: r0 ~ 1, interior radii spread, small center
    def make_init(scale=1.0, jit=0.0, seed=0):
        rng = np.random.default_rng(seed)
        r0 = 1.0 * scale
        ang_off = 0.0
        int_r = np.linspace(0.15, 0.75, 5) * scale + rng.normal(0, jit, 5)
        cr = 0.0 * scale
        ca = 0.0
        return np.array([r0, ang_off] + list(int_r) + [cr, ca])

    best_params = make_init(seed=0)
    best_val = -np.inf
    best_pts = None

    # Anneal beta from low to high for smooth -> sharp
    betas = [8.0, 20.0, 50.0, 100.0, 200.0]

    for beta in betas:
        if time.time() > _DEADLINE - 25:
            break
        # Multiple restarts from different scales/offsets to escape local minima
        for scale in [0.8, 1.0, 1.2]:
            for seed in [0, 1, 2]:
                if time.time() > _DEADLINE - 20:
                    break
                x0 = make_init(scale=scale, jit=0.02, seed=seed)
                # Bounds: r0 in (0.2, 2.0), ang_off in (-pi, pi), int_r in (0, 1.5), cr in (0,1.5), ca in (-pi,pi)
                lb = [0.2, -np.pi] + [0.0] * 5 + [0.0, -np.pi]
                ub = [2.0, np.pi] + [1.5] * 5 + [1.5, np.pi]
                try:
                    res = minimize(_neg_obj, x0, args=(beta,), method='L-BFGS-B',
                                   bounds=list(zip(lb, ub)),
                                   options={'maxiter': 200, 'maxfun': 4000, 'gtol': 1e-6})
                except Exception:
                    continue
                if res.success or res.fun < best_val * -1:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_params = res.x.copy()
                        best_pts = _params_to_points(res.x)

    # Final high-precision polish with exact (nonsmooth) objective via local Nelder-Mead on best region
    if best_pts is None:
        best_pts = _params_to_points(best_params)

    # Polish: use exact min-tri-area / hull with a fine local search
    def _exact_neg(params):
        pts = _params_to_points(params)
        soft = _min_tri_area(pts)
        hull = _hull_area(pts)
        if hull <= 1e-9:
            return 1e12
        return -(soft / hull)

    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_exact_neg, best_params, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if -res.fun > best_val:
                best_val = -res.fun
                best_params = res.x.copy()
                best_pts = _params_to_points(res.x)
        except Exception:
            pass

    # Normalize: scale so hull area = 1 (unit area), then return points
    pts = best_pts if best_pts is not None else _params_to_points(best_params)
    hull = _hull_area(pts)
    if hull > 1e-9:
        pts = pts / np.sqrt(hull)  # scaling by 1/sqrt(area) makes area -> 1
    return pts
