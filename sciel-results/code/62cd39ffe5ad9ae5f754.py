"""Heilbronn convex regions: maximize min triangle area over 13 points, normalized by hull area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 13
TIME_LIMIT = 115.0  # safety margin below 120s / 150s kill


def _min_triangle_area(pts):
    """Return the minimum triangle area over all triples of pts (n x 2 array)."""
    n = pts.shape[0]
    min_a = np.inf
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            v1 = pts[j] - pi
            for k in range(j + 1, n):
                v2 = pts[k] - pi
                a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if a < min_a:
                    min_a = a
                    if min_a < 1e-12:
                        return 0.0
    return min_a


def _hull_area(pts):
    """Area of the convex hull of pts."""
    from scipy.spatial import ConvexHull
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # Fallback: polygon area via shoelace on sorted-by-angle points
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        p = pts[order]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _score(pts):
    """min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha <= 0:
        return 0.0
    return _min_triangle_area(pts) / ha


def _convexity_violation(pts):
    """Sum of squared inward cross products if points are in CCW order.
    For a convex polygon traversed CCW, all cross products of consecutive
    edges are positive. We penalize negative ones."""
    n = pts.shape[0]
    if n < 3:
        return 0.0
    viol = 0.0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if cross < 0:
            viol += cross * cross
    return viol


def _objective(params, theta0):
    """Objective: -min_area + barrier for convexity and radial bounds."""
    r = params
    # radial bounds: keep within [0.5, 1.5]
    r_clip = np.clip(r, 0.3, 2.0)
    angles = theta0 + 0.0  # fixed angular spacing (regular polygon angles)
    pts = np.column_stack([r_clip * np.cos(angles), r_clip * np.sin(angles)])
    # ensure CCW order (already by construction)
    min_a = _min_triangle_area(pts)
    viol = _convexity_violation(pts)
    return -min_a + 1e3 * viol


def _refine_local(pts, time_budget):
    """Nelder-Mead style local refinement on 2D coordinates."""
    best = pts.copy()
    best_s = _score(best)
    start = time.time()
    # Use scipy Nelder-Mead on flattened coords
    def obj_flat(x):
        p = x.reshape(-1, 2)
        s = _score(p)
        return -s
    x0 = best.ravel()
    try:
        res = minimize(obj_flat, x0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            cs = _score(cand)
            if cs > best_s:
                best = cand
                best_s = cs
    except Exception:
        pass
    # A few more rounds of small perturbations if time remains
    while time.time() - start < time_budget:
        improved = False
        for _ in range(20):
            step = 0.01 * (0.5 ** (np.random.rand() * 3))
            cand = best + np.random.randn(*best.shape) * step
            cs = _score(cand)
            if cs > best_s:
                best = cand
                best_s = cs
                improved = True
        if not improved:
            break
    return best, best_s


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    def time_left():
        return deadline - time.time()

    # Start from regular 13-gon
    theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_pts = np.column_stack([np.cos(theta0), np.sin(theta0)])
    best_score = _score(best_pts)

    rng = np.random.default_rng(42)

    # Phase 1: radial-parameter optimization with SLSQP from several starts
    for trial in range(8):
        if time_left() < 25:
            break
        if trial == 0:
            r0 = np.ones(N)
        else:
            r0 = 1.0 + 0.15 * rng.standard_normal(N)
            r0 = np.clip(r0, 0.5, 1.5)
        try:
            res = minimize(_objective, r0, args=(theta0,), method='SLSQP',
                           bounds=[(0.3, 2.0)] * N,
                           options={'maxiter': 500, 'ftol': 1e-10})
            if res.x is not None:
                r = np.clip(res.x, 0.3, 2.0)
                pts = np.column_stack([r * np.cos(theta0), r * np.sin(theta0)])
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts
        except Exception:
            pass

    # Phase 2: local refinement on 2D coordinates (allows angular deviation)
    if time_left() > 10:
        best_pts, best_score = _refine_local(best_pts, max(0.0, time_left() - 15))

    # Phase 3: a couple of random restarts + refinement if time permits
    for _ in range(3):
        if time_left() < 20:
            break
        r0 = 1.0 + 0.2 * rng.standard_normal(N)
        r0 = np.clip(r0, 0.4, 1.6)
        # slight angular jitter too
        ang_jitter = 0.05 * rng.standard_normal(N)
        pts0 = np.column_stack([(r0 * np.cos(theta0 + ang_jitter)),
                                (r0 * np.sin(theta0 + ang_jitter))])
        # re-sort by angle to maintain convex order
        c = pts0.mean(axis=0)
        ang = np.arctan2(pts0[:, 1] - c[1], pts0[:, 0] - c[0])
        order = np.argsort(ang)
        pts0 = pts0[order]
        cand, s = _refine_local(pts0, max(0.0, time_left() - 15))
        if s > best_score:
            best_score = s
            best_pts = cand

    # Final safety: if we somehow produced a degenerate set, fall back to regular polygon
    if best_score <= 0 or not np.all(np.isfinite(best_pts)):
        best_pts = np.column_stack([np.cos(theta0), np.sin(theta0)])

    return best_pts
