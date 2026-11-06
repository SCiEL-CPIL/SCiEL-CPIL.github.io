import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping

_N = 8
_DEADLINE = 150.0          # evaluator wall-clock limit
_SAFETY = 15.0             # safety margin (seconds)
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _diameter(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return float(d.max())


def _log_product(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    return float(np.log(d).sum())


def _score_from_log_product_and_diameter(lp, dia):
    # score = Delta / n^n, Delta = prod_{ordered pairs} dist, after scaling diameter to 2.
    # scaling factor s = 2/dia; ordered-pair product scales by s^{n(n-1)}.
    n = _N
    return math.exp(lp + n * (n - 1) * (math.log(2.0) - math.log(dia)) - n * math.log(n))


def _score(pts):
    dia = _diameter(pts)
    if dia <= 1e-12:
        return 0.0
    return _score_from_log_product_and_diameter(_log_product(pts), dia)


# --- Reduced-space formulation ---
# Fix point 0 at origin, point 1 at (1, 0).  Free variables = coords of points 2..7 (12 floats).
# This removes translation, rotation, and reflection (D8) symmetries.
# The diameter is then max(1, max_{i>=2} |p_i|, max_{2<=i<j} |p_i - p_j|).
# Score in reduced space (diameter dia, log-product lp of all 8 points):
#   score = exp(lp + n(n-1)*(log 2 - log dia) - n log n)

def _reduced_to_points(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    pts = np.empty((_N, 2), dtype=float)
    pts[0] = (0.0, 0.0)
    pts[1] = (1.0, 0.0)
    pts[2:] = x.reshape(_N - 2, 2)
    return pts


def _reduced_diameter(pts):
    # diameter = max distance among all pairs (point 0 at origin, point 1 at (1,0))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return float(d.max())


def _reduced_objective(x):
    pts = _reduced_to_points(x)
    dia = _reduced_diameter(pts)
    if dia < 1e-8:
        return 1e12
    # log product of all pairwise distances (8 points, 28 unordered pairs -> use full matrix)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    lp = float(np.log(d).sum())
    n = _N
    # maximize score <=> minimize negative log score
    log_score = lp + n * (n - 1) * (math.log(2.0) - math.log(dia)) - n * math.log(n)
    return -log_score


def _reduced_score(x):
    pts = _reduced_to_points(x)
    dia = _reduced_diameter(pts)
    if dia < 1e-8:
        return 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    lp = float(np.log(d).sum())
    n = _N
    return math.exp(lp + n * (n - 1) * (math.log(2.0) - math.log(dia)) - n * math.log(n))


def _refine_reduced(x0):
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    best_x = x0.copy()
    best_score = _reduced_score(x0)

    # Stage 1: L-BFGS-B
    try:
        res = minimize(_reduced_objective, x0, method='L-BFGS-B',
                       options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
        cand = res.x
        s = _reduced_score(cand)
        if s > best_score:
            best_x, best_score = cand, s
    except Exception:
        pass

    # Stage 2: Tight L-BFGS-B
    if _remaining() > 5.0:
        try:
            res = minimize(_reduced_objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
            cand = res.x
            s = _reduced_score(cand)
            if s > best_score:
                best_x, best_score = cand, s
        except Exception:
            pass

    # Stage 3: SLSQP
    if _remaining() > 8.0:
        try:
            res = minimize(_reduced_objective, best_x, method='SLSQP',
                           options={'maxiter': 500, 'ftol': 1e-14})
            cand = res.x
            s = _reduced_score(cand)
            if s > best_score:
                best_x, best_score = cand, s
        except Exception:
            pass

    return best_x, best_score


def _make_cage_reduced(kind, rng):
    """Return reduced-space variable vector (12 floats) for a given cage topology."""
    if kind == 0:  # concentric squares
        a = 1.0
        b = 0.5
        pts = np.zeros((_N, 2))
        pts[0] = (0.0, 0.0)
        pts[1] = (1.0, 0.0)
        # Place remaining 6 points: 3 on outer circle (radius a), 3 on inner (radius b)
        # Outer: angles offset from 0
        for k in range(3):
            ang = math.pi / 2 + 2 * math.pi * k / 3  # spread around
            pts[2 + k] = (a * math.cos(ang), a * math.sin(ang))
        for k in range(3):
            ang = math.pi / 3 + 2 * math.pi * k / 3
            pts[5 + k] = (b * math.cos(ang), b * math.sin(ang))
        return pts[2:].ravel()

    if kind == 1:  # nested rings (outer 4, inner 2 besides the fixed 2)
        pts = np.zeros((_N, 2))
        pts[0] = (0.0, 0.0)
        pts[1] = (1.0, 0.0)
        # 4 outer at radius 1, 2 inner at radius 0.5
        for k in range(4):
            ang = math.pi / 4 + 2 * math.pi * k / 4
            pts[2 + k] = (1.0 * math.cos(ang), 1.0 * math.sin(ang))
        for k in range(2):
            ang = math.pi / 2 + math.pi * k
            pts[6 + k] = (0.5 * math.cos(ang), 0.5 * math.sin(ang))
        return pts[2:].ravel()

    if kind == 2:  # deformed octagon (regular octagon centered, then shift so p0=origin, p1=(1,0))
        pts = np.zeros((_N, 2))
        for k in range(_N):
            ang = 2 * math.pi * k / _N
            r = 1.0 + 0.15 * rng.uniform(-1, 1)
            pts[k] = (r * math.cos(ang), r * math.sin(ang))
        # Center it
        pts -= pts.mean(axis=0)
        # Now rotate/translate so that the two points with max distance are at origin and (1,0)
        # Find the diameter pair
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, 0)
        idx = np.unravel_index(np.argmax(d), d.shape)
        i, j = idx
        p0, p1 = pts[i], pts[j]
        vec = p1 - p0
        length = np.linalg.norm(vec)
        # Rotate so vec is along +x, and translate so p0 is at origin
        cos_t = vec[0] / length
        sin_t = vec[1] / length
        # Rotation matrix to map vec to (1,0): R = [[cos, sin], [-sin, cos]]
        R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        pts = (pts - p0) @ R.T
        pts[0] = (0.0, 0.0)
        pts[1] = (length, 0.0)
        # We need diameter = length, so this is already the diameter pair.
        # Return reduced variables (points 2..7)
        return pts[2:].ravel()

    if kind == 3:  # random
        return rng.uniform(-1.5, 1.5, size=(_N - 2, 2)).ravel()

    # kind 4: regular octagon in reduced form
    pts = np.zeros((_N, 2))
    for k in range(_N):
        ang = 2 * math.pi * k / _N
        pts[k] = (math.cos(ang), math.sin(ang))
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 0)
    idx = np.unravel_index(np.argmax(d), d.shape)
    i, j = idx
    p0, p1 = pts[i], pts[j]
    vec = p1 - p0
    length = np.linalg.norm(vec)
    cos_t = vec[0] / length
    sin_t = vec[1] / length
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    pts = (pts - p0) @ R.T
    pts[0] = (0.0, 0.0)
    pts[1] = (length, 0.0)
    return pts[2:].ravel()

    # kind 5: square + 4 extra points
    pts = np.zeros((_N, 2))
    pts[0] = (0.0, 0.0)
    pts[1] = (1.0, 0.0)
    # 4 corners of a square around center (0.5, 0)
    cx, cy = 0.5, 0.0
    r = 0.5
    for k in range(4):
        ang = math.pi / 4 + 2 * math.pi * k / 4
        pts[2 + k] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    # 2 extra points
    pts[6] = (0.5, 0.7)
    pts[7] = (0.5, -0.7)
    return pts[2:].ravel()


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # Start with regular octagon in reduced form
    best_x = _make_cage_reduced(4, rng)
    best_score = _reduced_score(best_x)

    # 1) Local refinement from diverse combinatorial topologies
    for kind in range(6):
        if _remaining() < 2.0:
            break
        init = _make_cage_reduced(kind, rng)
        cand_x, s = _refine_reduced(init)
        if s > best_score:
            best_x, best_score = cand_x, s

    # 2) Additional random starts with refinement
    n_random_starts = 15
    for i in range(n_random_starts):
        if _remaining() < 4.0:
            break
        init = _make_cage_reduced(3, rng)
        cand_x, s = _refine_reduced(init)
        if s > best_score:
            best_x, best_score = cand_x, s

    # 3) Global population-based exploration (differential evolution) in reduced space
    bounds = [(-2.0, 2.0)] * (2 * (_N - 2))
    if _remaining() > 40.0:
        try:
            de_res = differential_evolution(
                _reduced_objective, bounds,
                maxiter=100, popsize=25, seed=12345, tol=1e-10,
                polish=True, mutation=(0.5, 1.0), recombination=0.8,
                update_interval=1,
            )
            cand_x = np.asarray(de_res.x, dtype=float).reshape(-1)
            cand_x, s = _refine_reduced(cand_x)
            if s > best_score:
                best_x, best_score = cand_x, s
        except Exception:
            pass

    # 4) Basin Hopping on the best candidate found so far
    if _remaining() > 15.0:
        try:
            bh_res = basinhopping(
                _reduced_objective, best_x,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 200, 'ftol': 1e-12}},
                niter=80, seed=12345,
                stepsize=0.4
            )
            cand_x = bh_res.x
            cand_x, s = _refine_reduced(cand_x)
            if s > best_score:
                best_x, best_score = cand_x, s
        except Exception:
            pass

    # 5) More random starts if time permits
    while _remaining() > 6.0:
        init = _make_cage_reduced(3, rng)
        cand_x, s = _refine_reduced(init)
        if s > best_score:
            best_x, best_score = cand_x, s

    # Final high-precision refinement of the best configuration
    if _remaining() > 3.0:
        cand_x, s = _refine_reduced(best_x)
        if s > best_score:
            best_x, best_score = cand_x, s

    # Convert reduced solution back to full 8-point configuration
    best = _reduced_to_points(best_x)

    # Ensure distinct points and bounded coordinates
    for i in range(_N):
        for j in range(i + 1, _N):
            if np.linalg.norm(best[i] - best[j]) < 1e-6:
                best[j] += [1e-6, 0]

    # Clamp coordinates
    best = np.clip(best, -1e6, 1e6)

    return [[float(x), float(y)] for (x, y) in best]
