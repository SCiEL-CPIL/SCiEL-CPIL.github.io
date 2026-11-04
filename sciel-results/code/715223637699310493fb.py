import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14
N_OUT = 7
N_IN = 7

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])
_TI0 = TRIPLE_INDICES[:, 0]
_TI1 = TRIPLE_INDICES[:, 1]
_TI2 = TRIPLE_INDICES[:, 2]

# Precompute the 7 outer unit-circle angles
_ANG_OUT = np.linspace(0.0, 2.0 * np.pi, N_OUT, endpoint=False)
_COS_OUT = np.cos(_ANG_OUT)
_SIN_OUT = np.sin(_ANG_OUT)

# Precompute the 7 inner unit-circle angles (base, before rotation)
_ANG_IN_BASE = np.linspace(0.0, 2.0 * np.pi, N_IN, endpoint=False)

# Precompute cross-product coefficients for the mixed (1,2) and (2,1) triangle families.
# For each pair of inner indices (a,b) and each outer index c, the area of triangle
# (in_a, in_b, out_c) is 0.5*r*| (cos_b - cos_a)*sin_c - (sin_b - sin_a)*cos_c |.
_MIXED_AB = np.zeros((N_OUT, N_IN, N_IN))   # coefficient of sin_c
_MIXED_BS = np.zeros((N_OUT, N_IN, N_IN))   # coefficient of cos_c (with sign)
for c in range(N_OUT):
    sc = _SIN_OUT[c]
    cc = _COS_OUT[c]
    for a in range(N_IN):
        for b in range(N_IN):
            if a == b:
                continue
            _MIXED_AB[c, a, b] = _COS_OUT[b] - _COS_OUT[a]
            _MIXED_BS[c, a, b] = _SIN_OUT[b] - _SIN_OUT[a]

# Mixed (2,1): triangle (in_a, out_b, out_c). Area = 0.5*| r*(sin_b - sin_c)*cos_a - r*(cos_b - cos_c)*sin_a |
_MIXED_BC = np.zeros((N_OUT, N_OUT, N_IN))   # coefficient of cos_a
_MIXED_CS = np.zeros((N_OUT, N_OUT, N_IN))   # coefficient of sin_a
for b in range(N_OUT):
    for c in range(N_OUT):
        if b == c:
            continue
        sb = _SIN_OUT[b]
        sc = _SIN_OUT[c]
        cb = _COS_OUT[b]
        cc = _COS_OUT[c]
        for a in range(N_IN):
            ca = _COS_OUT[a]
            sa = _SIN_OUT[a]
            _MIXED_BC[b, c, a] = (sb - sc) * ca
            _MIXED_CS[b, c, a] = (cb - cc) * sa


def _pure_areas(r, theta):
    """Return the 7 pure outer and 7 pure inner triangle areas (each 0.5*|cross|)."""
    # Pure outer: regular heptagon on unit circle, area of triangle (i, i+1, i+2)
    # = 0.5 * |sin(2*pi/7) * 2| ... compute directly.
    # Area of triangle formed by three consecutive outer points.
    # Using vectors: for points at angles 0, d, 2d, area = 0.5*| (p2-p0) x (p1-p0) |
    d = 2.0 * np.pi / N_OUT
    # area = 0.5 * | (cos(2d)-cos0)*(sin d - 0) - (sin(2d)-0)*(cos d - 0) |
    pure_outer = 0.5 * np.abs((_COS_OUT[2] - _COS_OUT[0]) * (_SIN_OUT[1] - _SIN_OUT[0])
                              - (_SIN_OUT[2] - _SIN_OUT[0]) * (_COS_OUT[1] - _COS_OUT[0]))
    # Pure inner: same shape scaled by r^2
    pure_inner = r * r * pure_outer
    return pure_outer, pure_inner


def _build_points(r, theta):
    ang_in = _ANG_IN_BASE + theta
    x_in = r * np.cos(ang_in)
    y_in = r * np.sin(ang_in)
    return np.vstack([
        np.column_stack((_COS_OUT, _SIN_OUT)),
        np.column_stack((x_in, y_in)),
    ])


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(points):
    p_i = points[_TI0]
    p_j = points[_TI1]
    p_k = points[_TI2]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    return 0.5 * np.min(np.abs(cross))


def _calculate_score(points):
    """Score: min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return _min_triangle_area(points) / hull_area


def _soft_min_score(points, beta=50.0):
    """Soft-min surrogate: soft_min(triangle areas) / hull area."""
    p_i = points[_TI0]
    p_j = points[_TI1]
    p_k = points[_TI2]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta
    hull_area = _hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return soft_min / hull_area


def _ring_score_fast(r, theta):
    """Exact score for the 7-7 ring, computed with precomputed coefficients (fast)."""
    if r < 0.0 or r > 0.9999:
        return 0.0
    # Pure areas (scalar)
    pure_outer, pure_inner = _pure_areas(r, theta)

    # Mixed (1,2): for each outer c, inner pair (a,b): 0.5*r*| MIXED_AB[c,a,b]*sin(theta+ang_c) - MIXED_BS[c,a,b]*cos(theta+ang_c) |
    # Precompute per-outer sin/cos of (theta + ang_c)
    sin_c = np.sin(theta + _ANG_OUT)
    cos_c = np.cos(theta + _ANG_OUT)

    # Build a 2D array of the linear expression for (1,2): shape (N_OUT, N_IN*N_IN)
    # term = MIXED_AB * sin_c[:,None,None] - MIXED_BS * cos_c[:,None,None]
    term12 = _MIXED_AB * sin_c[:, None, None] - _MIXED_BS * cos_c[:, None, None]
    areas12 = 0.5 * r * np.abs(term12)

    # Mixed (2,1): for each outer pair (b,c), inner a: 0.5*| r*( MIXED_BC[b,c,a]*cos(theta+ang_a) - MIXED_CS[b,c,a]*sin(theta+ang_a) ) |
    sin_a = np.sin(theta + _ANG_IN_BASE)
    cos_a = np.cos(theta + _ANG_IN_BASE)
    term21 = _MIXED_BC * cos_a[None, None, :] - _MIXED_CS * sin_a[None, None, :]
    areas21 = 0.5 * np.abs(r * term21)

    min_area = min(pure_outer, pure_inner, np.min(areas12), np.min(areas21))

    # Hull area: points on unit circle -> hull is regular heptagon (outer ring)
    hull_area = 0.5 * N_OUT * np.sin(2.0 * np.pi / N_OUT)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _ring_soft_score_fast(r, theta, beta=50.0):
    """Soft-min surrogate for the 7-7 ring using precomputed coefficients."""
    if r < 0.0 or r > 0.9999:
        return 0.0
    pure_outer, pure_inner = _pure_areas(r, theta)

    sin_c = np.sin(theta + _ANG_OUT)
    cos_c = np.cos(theta + _ANG_OUT)
    term12 = _MIXED_AB * sin_c[:, None, None] - _MIXED_BS * cos_c[:, None, None]
    areas12 = 0.5 * r * np.abs(term12)

    sin_a = np.sin(theta + _ANG_IN_BASE)
    cos_a = np.cos(theta + _ANG_IN_BASE)
    term21 = _MIXED_BC * cos_a[None, None, :] - _MIXED_CS * sin_a[None, None, :]
    areas21 = 0.5 * np.abs(r * term21)

    all_areas = np.concatenate([
        np.array([pure_outer]), np.array([pure_inner]),
        areas12.ravel(), areas21.ravel(),
    ])
    max_area = np.max(all_areas)
    shifted = -beta * (all_areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta
    hull_area = 0.5 * N_OUT * np.sin(2.0 * np.pi / N_OUT)
    return soft_min / hull_area


def _optimize_77(time_budget):
    """Optimize the 7-7 two-ring configuration over (r, theta)."""
    start = time.time()
    deadline = start + time_budget

    # Symmetry: configuration invariant under rotation by 2*pi/7 for theta.
    theta_max = 2.0 * np.pi / N_OUT

    # Stage 1: coarse 2D grid to find a good basin.
    best_score = -np.inf
    best_params = None
    n_r = 60
    n_th = 40
    r_grid = np.linspace(0.05, 0.95, n_r)
    th_grid = np.linspace(0.0, theta_max, n_th)
    for r in r_grid:
        if time.time() > deadline:
            break
        for th in th_grid:
            s = _ring_score_fast(r, th)
            if s > best_score:
                best_score = s
                best_params = (r, th)

    # Stage 2: local Nelder-Mead polish from the best grid point and a few neighbors.
    if best_params is not None and time.time() < deadline - 1.0:
        r0, th0 = best_params
        # Candidate starts: best + perturbations
        starts = [(r0, th0)]
        for dr in (-0.05, 0.05):
            for dth in (-0.2, 0.2):
                rr = r0 + dr
                tt = (th0 + dth) % theta_max
                if 0.01 <= rr <= 0.99:
                    starts.append((rr, tt))
        for (rr, tt) in starts:
            if time.time() > deadline - 0.5:
                break
            try:
                res = minimize(
                    lambda p: -_ring_score_fast(p[0], p[1]),
                    np.array([rr, tt]),
                    method='Nelder-Mead',
                    bounds=[(0.01, 0.99), (0.0, theta_max)],
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000,
                             'adaptive': True},
                )
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_params = (res.x[0], res.x[1] % theta_max)
            except Exception:
                pass

    # Stage 3: high-precision polish with the smooth soft-min surrogate (increasing beta).
    if best_params is not None and time.time() < deadline - 0.5:
        current = np.array(best_params, dtype=float)
        for beta in [20.0, 50.0, 100.0, 200.0, 400.0]:
            if time.time() > deadline - 0.3:
                break
            try:
                res = minimize(
                    lambda p: -_ring_soft_score_fast(p[0], p[1], beta),
                    current,
                    method='L-BFGS-B',
                    bounds=[(0.01, 0.99), (0.0, theta_max)],
                    options={'ftol': 1e-14, 'gtol': 1e-12, 'maxiter': 2000},
                )
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_params = (res.x[0], res.x[1] % theta_max)
                current = res.x
            except Exception:
                pass

    # Stage 4: final exact re-evaluation and a tiny Nelder-Mead micro-polish on exact score.
    if best_params is not None and time.time() < deadline - 0.2:
        r0, th0 = best_params
        try:
            res = minimize(
                lambda p: -_ring_score_fast(p[0], p[1]),
                np.array([r0, th0]),
                method='Nelder-Mead',
                bounds=[(0.01, 0.99), (0.0, theta_max)],
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000,
                         'adaptive': True},
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_params = (res.x[0], res.x[1] % theta_max)
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
    return best_score, best_params


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    score, params = _optimize_77(deadline - start_time - 5.0)

    if params is None:
        # Fallback: a reasonable default 7-7 configuration
        r, theta = 0.5, 0.0
    else:
        r, theta = params

    points = _build_points(r, theta)

    # Verify with the exact evaluator path; if it fails, fall back to default.
    s = _calculate_score(points)
    if s <= 0.0:
        r, theta = 0.5, 0.0
        points = _build_points(r, theta)

    return points
