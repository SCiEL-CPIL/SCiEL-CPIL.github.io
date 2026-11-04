"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points.

Strategy:
  1. Generate diverse initial points (regular polygon, jittered, random convex, Halton).
  2. Global search via differential evolution in Cartesian space with a smooth objective.
  3. Local refinement via L-BFGS-B on a soft-min surrogate.
  4. Final polish with Nelder-Mead on the exact objective.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution, NelderMead
from itertools import combinations

N = 13
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])


def _tri_areas(pts):
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def _score(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return float(np.min(_tri_areas(pts)) / ha)


def _soft_min(areas, tau):
    max_a = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_a) / tau)))
    return max_a - tau * log_sum_exp


def _neg_soft_min_ratio(pts, temp):
    pts = np.asarray(pts, dtype=np.float64)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e8
    a = _tri_areas(pts)
    a = np.maximum(a, 1e-15)
    s = -a / temp
    s = s - np.max(s)
    logsum = np.log(np.sum(np.exp(s)))
    soft_min = -logsum * temp
    return soft_min / ha


def _regular_polygon():
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(th), np.sin(th)])


def _jittered_polygon(scale=0.05, seed=0):
    rng = np.random.RandomState(seed)
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + scale * rng.randn(N)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def _random_convex(seed=0):
    rng = np.random.RandomState(seed)
    th = np.sort(rng.uniform(0, 2 * np.pi, N))
    r = rng.uniform(0.5, 1.0, N)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def _halton_points(seed=0):
    # Halton sequence in 2D, scaled to unit disk
    def halton(n, base):
        f = 1.0
        r = 0.0
        while n > 0:
            f /= base
            r += f * (n % base)
            n //= base
        return r
    pts = np.array([[halton(i + 1, 2), halton(i + 1, 3)] for i in range(N)])
    pts = 2 * pts - 1  # map to [-1,1]^2
    # Ensure convex position by taking convex hull and if fewer than N, add interior
    # For simplicity, just use as-is; DE will fix
    return pts


def _refine_cartesian(pts, time_budget):
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)

    # First pass: moderate temp
    for temp in [0.01, 0.003, 0.001]:
        if time.time() - t0 > time_budget - 2:
            break
        def obj(x):
            p = x.reshape(N, 2)
            return _neg_soft_min_ratio(p, temp)
        try:
            res = minimize(obj, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-9})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    # Nelder-Mead polish on exact objective
    if time.time() - t0 < time_budget - 1:
        def exact_obj(x):
            p = x.reshape(N, 2)
            return -_score(p)
        try:
            res = minimize(exact_obj, best.flatten(), method='Nelder-Mead',
                           options={'maxiter': 300, 'fatol': 1e-10, 'xatol': 1e-10})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    return best, best_s


def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    # Initial candidates
    candidates = []
    candidates.append(_regular_polygon())
    for s in range(5):
        candidates.append(_jittered_polygon(scale=0.05 + 0.02 * s, seed=s))
    for s in range(3):
        candidates.append(_random_convex(seed=s))
    candidates.append(_halton_points())

    best_pts = candidates[0].copy()
    best_score = _score(candidates[0])
    for c in candidates[1:]:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_pts = c.copy()

    # Global search: DE in Cartesian space
    if time.time() < deadline - 60:
        def de_obj(params):
            if time.time() > deadline - 5:
                return 1e10
            pts = params.reshape(N, 2)
            ha = _hull_area(pts)
            if ha < 1e-10:
                return 1e10
            a = _tri_areas(pts)
            if np.any(a < 1e-12):
                return 1e10
            smooth_min = _soft_min(a, 0.02)
            return -smooth_min / ha

        bounds = [(-1.5, 1.5)] * (2 * N)
        try:
            res_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            cand = res_de.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
        except Exception:
            pass

    # Local refinement
    remaining = deadline - time.time() - 2
    if remaining > 10:
        try:
            pts, s = _refine_cartesian(best_pts, remaining)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass

    # Multi-start polish from best
    if time.time() < deadline - 5:
        rng = np.random.RandomState(123)
        for trial in range(5):
            if time.time() > deadline - 3:
                break
            perturbed = best_pts + 0.01 * rng.randn(N, 2)
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_pts = perturbed.copy()
            # Quick local refine
            remaining2 = deadline - time.time() - 2
            if remaining2 > 5:
                try:
                    pts, s2 = _refine_cartesian(perturbed, remaining2)
                    if s2 > best_score:
                        best_score = s2
                        best_pts = pts.copy()
                except Exception:
                    pass

    return best_pts
