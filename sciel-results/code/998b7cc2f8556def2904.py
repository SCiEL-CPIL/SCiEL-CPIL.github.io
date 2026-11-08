"""Heilbronn convex regions - 13 points maximizing min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
# All triples (i,j,k) with i<j<k
import itertools
_TRIPLES = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int64)
_I = _TRIPLES[:, 0]
_J = _TRIPLES[:, 1]
_K = _TRIPLES[:, 2]
_NTRIP = _TRIPLES.shape[0]


def _hull_area(pts):
    """Convex hull area via shoelace on hull vertices. Returns 0 on failure."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume == area
    except Exception:
        return 0.0


def _all_triangle_areas(pts):
    """Vectorized areas of all C(n,3) triangles. Shape (ntrip,)."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _exact_score(pts):
    """min triangle area / convex hull area."""
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(pts)
    return float(areas.min() / hull_area)


def _softmin_objective(x, beta):
    """
    Objective to MINIMIZE: -softmin(min_triangle_area / hull_area).
    x is flat 26-vector of point coordinates.
    Uses a smooth soft-min over triangle areas, normalized by hull area.
    """
    pts = x.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(pts)
    # softmin of areas / hull_area
    a = areas / hull_area
    # numerically stable softmin
    m = a.min()
    # softmin = m - (1/beta)*log(sum(exp(-beta*(a-m))))
    s = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (a - m))))
    return -s


def _regular_polygon(n, rot=0.0, r=1.0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
    return r * np.column_stack([np.cos(ang), np.sin(ang)])


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill

    best_score = -1.0
    best_points = None

    def consider(pts):
        nonlocal best_score, best_points
        s = _exact_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # ---- Phase 0: evaluate structured initializations ----
    init_pool = []
    # Regular 13-gon (canonical)
    init_pool.append(_regular_polygon(N))
    # Rotated regular polygons
    for rot in [0.1, 0.3, 0.5, 0.7, 1.0]:
        init_pool.append(_regular_polygon(N, rot=rot))
    # Regular polygon on a circle of different radius (scale doesn't matter after norm,
    # but helps L-BFGS start in a non-degenerate region)
    init_pool.append(_regular_polygon(N, r=1.5))
    # Ellipse-based (still convex hull = ellipse boundary points)
    for ax, by in [(1.0, 0.85), (1.15, 0.9), (0.9, 1.1), (1.3, 0.7)]:
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        init_pool.append(np.column_stack([ax * np.cos(ang), by * np.sin(ang)]))
    # Perturbed regular polygons (deterministic)
    rng = np.random.RandomState(7)
    for trial in range(6):
        base = _regular_polygon(N)
        ang = np.arctan2(base[:, 1], base[:, 0])
        rad = np.linalg.norm(base, axis=1)
        ang_p = np.sort(ang + 0.05 * rng.randn(N))
        rad_p = rad + 0.05 * rng.randn(N)
        rad_p = np.clip(rad_p, 0.3, 2.0)
        init_pool.append(np.column_stack([rad_p * np.cos(ang_p), rad_p * np.sin(ang_p)]))

    for pts in init_pool:
        if time.time() > deadline - 5:
            break
        consider(pts)

    if best_points is None:
        best_points = _regular_polygon(N)
        best_score = _exact_score(best_points)

    # ---- Phase 1: L-BFGS-B refinement with a beta schedule ----
    # Start at moderate beta (smooth), ramp up to large beta (sharp -> true min).
    beta_schedule = [8.0, 20.0, 50.0, 120.0, 300.0, 700.0, 1500.0]

    # We'll refine from the current best, and also from a few good initializations.
    # Build a small set of starting points for multi-start local optimization.
    start_set = [best_points.copy()]
    # Add the best few from the pool (distinct)
    scored = []
    for pts in init_pool:
        scored.append((_exact_score(pts), pts))
    scored.sort(key=lambda t: -t[0])
    for s, pts in scored[:4]:
        start_set.append(pts.copy())
    # Remove duplicates by rounding
    uniq = []
    for p in start_set:
        key = np.round(p, 3)
        if not any(np.array_equal(key, np.round(u, 3)) for u in uniq):
            uniq.append(p)
    start_set = uniq

    for s_idx, x0 in enumerate(start_set):
        if time.time() > deadline - 30:
            break
        x_cur = x0.flatten().copy()
        for beta in beta_schedule:
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(
                    _softmin_objective,
                    x_cur,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50},
                )
                x_cur = res.x
                # Track best exact score along the way
                pts_now = x_cur.reshape(N, 2)
                consider(pts_now)
                if time.time() > deadline - 20:
                    break
            except Exception:
                pass
        # After the schedule, do a final high-beta polish
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                _softmin_objective,
                x_cur,
                args=(3000.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50},
            )
            x_cur = res.x
            consider(x_cur.reshape(N, 2))
        except Exception:
            pass

    # ---- Phase 2: coordinate-descent style fine polish on the best ----
    # A direct, derivative-free local search that nudge-optimizes each coordinate
    # while holding others fixed, using the EXACT objective. This escapes the
    # soft-min approximation bias near the end.
    if best_points is not None and time.time() < deadline - 8:
        def exact_neg(x):
            return -_exact_score(x.reshape(N, 2))

        x = best_points.flatten().copy()
        # Adaptive step sizes per coordinate, geometric reduction.
        base_steps = np.full(2 * N, 1e-2)
        for outer in range(40):
            if time.time() > deadline - 6:
                break
            improved = False
            for d in range(2 * N):
                if time.time() > deadline - 6:
                    break
                step = base_steps[d]
                x0 = x[d]
                cur = -_exact_score(x.reshape(N, 2))
                # Try a small set of offsets around current
                best_d = x0
                best_val = cur
                for mult in [1.0, 0.5, 0.25, 2.0, 0.1]:
                    for sign in [-1, 1]:
                        x[d] = x0 + sign * step * mult
                        v = -_exact_score(x.reshape(N, 2))
                        if v < best_val - 1e-12:
                            best_val = v
                            best_d = x[d]
                if best_d != x0:
                    x[d] = best_d
                    improved = True
                    base_steps[d] *= 1.5  # allow bigger moves if improving
                    base_steps[d] = min(base_steps[d], 5e-2)
                else:
                    base_steps[d] *= 0.6  # shrink
            if not improved:
                # All steps shrunk; if steps are tiny, stop
                if np.max(base_steps) < 1e-5:
                    break
        consider(x.reshape(N, 2))

    # ---- Final safety ----
    if best_points is None:
        best_points = _regular_polygon(N)

    # Ensure the returned set has a valid hull and is finite
    if not np.all(np.isfinite(best_points)) or _hull_area(best_points) <= 1e-12:
        best_points = _regular_polygon(N)

    return best_points
