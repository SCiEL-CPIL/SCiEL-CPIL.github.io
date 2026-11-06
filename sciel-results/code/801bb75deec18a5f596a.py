"""Suite task: 8 planar points maximizing product of pairwise distances at fixed diameter.
Structural change: multi-start global search (differential evolution + diverse
combinatorial/random topologies) each refined by L-BFGS-B, with a strict deadline guard.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 8
_DEADLINE = 150.0          # evaluator wall-clock limit
_SAFETY = 12.0             # safety margin (seconds)
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
    return math.exp(lp - n * (n - 1) * math.log(dia) + n * (n - 1) * math.log(2.0)) / (n ** n)


def _score(pts):
    dia = _diameter(pts)
    if dia <= 1e-12:
        return 0.0
    return _score_from_log_product_and_diameter(_log_product(pts), dia)


def _objective(pts):
    # maximize score  <=>  minimize -score.  Guard against degenerate diameters.
    dia = _diameter(pts)
    if dia < 1e-8:
        return 1e12
    return -_score_from_log_product_and_diameter(_log_product(pts), dia)


def _refine(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    try:
        res = minimize(_objective, pts.ravel(), method='L-BFGS-B')
        cand = res.x.reshape(_N, 2)
        s = _score(cand)
        if s > _score(pts):
            return cand, s
    except Exception:
        pass
    return pts, _score(pts)


def _make_cage_init(kind, rng):
    if kind == 0:  # concentric squares
        a = 1.0
        b = 0.5
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts.append([a * math.cos(ang), a * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([b * math.cos(ang), b * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 1:  # nested rings (outer 5, inner 3)
        pts = []
        for k in range(5):
            ang = 2 * math.pi * k / 5
            pts.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
        for k in range(3):
            ang = 2 * math.pi * k / 3
            pts.append([0.5 * math.cos(ang), 0.5 * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 2:  # deformed octagon
        pts = []
        for k in range(_N):
            ang = 2 * math.pi * k / _N
            r = 1.0 + 0.15 * rng.uniform(-1, 1)
            pts.append([r * math.cos(ang), r * math.sin(ang)])
        return np.array(pts, dtype=float)
    # kind 3: random
    return rng.uniform(-1.0, 1.0, size=(8, 2))


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # Seed with the regular octagon.
    oct_pts = np.array([[math.cos(2 * math.pi * k / _N),
                         math.sin(2 * math.pi * k / _N)] for k in range(_N)], dtype=float)
    best, best_score = oct_pts, _score(oct_pts)

    # 1) Local refinement from diverse combinatorial topologies (cheap, high value).
    for kind in range(4):
        if _remaining() < 2.0:
            break
        init = _make_cage_init(kind, rng)
        cand, s = _refine(init)
        if s > best_score:
            best, best_score = cand, s

    # 2) Global population-based exploration (differential evolution) if budget allows.
    bounds = [(-1.0, 1.0)] * (2 * _N)
    if _remaining() > 40.0:
        try:
            de_res = differential_evolution(
                _objective, bounds,
                maxiter=60, popsize=15, seed=12345, tol=1e-8,
                polish=True, mutation=(0.4, 1.0), recombination=0.8,
                update_interval=1,
            )
            cand = np.asarray(de_res.x, dtype=float).reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 3) Extra multi-start random + local refinement while budget remains.
    while _remaining() > 6.0:
        init = _make_cage_init(3, rng)
        cand, s = _refine(init)
        if s > best_score:
            best, best_score = cand, s

    # Final high-precision refinement of the best configuration.
    if _remaining() > 2.0:
        cand, s = _refine(best)
        if s > best_score:
            best, best_score = cand, s

    best = np.asarray(best, dtype=float).reshape(_N, 2)
    # Ensure distinct points and bounded coordinates.
    return [[float(x), float(y)] for (x, y) in best]
