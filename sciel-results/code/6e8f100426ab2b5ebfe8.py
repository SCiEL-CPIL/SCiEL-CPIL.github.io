"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 14

# Precompute all triples
_TRIPLES = list(combinations(range(N), 3))
_TIDX = np.array(_TRIPLES, dtype=np.int32)  # (364, 3)
_NTRIP = len(_TRIPLES)

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Minimum area over all triples (364)."""
    p = np.asarray(points, dtype=np.float64)
    a = p[_TIDX[:, 0]]
    b = p[_TIDX[:, 1]]
    c = p[_TIDX[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    return float(areas.min())


def _convex_hull_area(points):
    p = np.asarray(points, dtype=np.float64)
    n = p.shape[0]
    if n < 3:
        return 0.0
    idx = np.arange(n)
    # Sort by (x, y)
    order = np.lexsort((p[:, 1], p[:, 0]))
    p = p[order]
    # Andrew's monotone chain
    lower = []
    for i in range(n):
        while len(lower) >= 2:
            o = lower[-2]
            l = lower[-1]
            cr = (p[l, 0] - p[o, 0]) * (p[i, 1] - p[o, 1]) - (p[l, 1] - p[o, 1]) * (p[i, 0] - p[o, 0])
            if cr <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(i)
    upper = []
    for i in range(n - 1, -1, -1):
        while len(upper) >= 2:
            o = upper[-2]
            l = upper[-1]
            cr = (p[l, 0] - p[o, 0]) * (p[i, 1] - p[o, 1]) - (p[l, 1] - p[o, 1]) * (p[i, 0] - p[o, 0])
            if cr <= 1e-12:
                upper.pop()
            else:
                break
        upper.append(i)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    h = p[hull]
    x = h[:, 0]
    y = h[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def _score(points):
    m = _min_triangle_area(points)
    h = _convex_hull_area(points)
    if h <= 1e-12:
        return 0.0
    return m / h


def _smooth_obj(x, beta):
    """Log-sum-exp surrogate of negative triangle areas (smooth max)."""
    p = x.reshape(N, 2)
    a = p[_TIDX[:, 0]]
    b = p[_TIDX[:, 1]]
    c = p[_TIDX[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    neg = -areas
    m = neg.max()
    if not np.isfinite(m):
        return 1e9
    s = np.exp(beta * (neg - m))
    return m + np.log(s.sum()) / beta


def _regular_polygon(n, cx=0.5, cy=0.5, r=0.5, phase=0.0):
    t = np.arange(n) * (2.0 * np.pi / n) + phase
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _local_polish(points, deadline, step=0.02, iters=4000):
    """Exact min-triangle-area local search with adaptive step."""
    best = points.copy()
    best_val = _min_triangle_area(best)
    step = float(step)
    for it in range(iters):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for dx in (-step, 0.0, step):
                for dy in (-step, 0.0, step):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    cand = best.copy()
                    cand[i, 0] += dx
                    cand[i, 1] += dy
                    v = _min_triangle_area(cand)
                    if v > best_val:
                        best = cand
                        best_val = v
                        improved = True
        if not improved:
            step *= 0.6
            if step < 1e-6:
                break
    return best


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s

    # Initial candidate: regular 14-gon centered in [0,1]^2
    base = _regular_polygon(N, cx=0.5, cy=0.5, r=0.5, phase=0.0)

    best = base.copy()
    best_score = _score(best)

    # Try a few phase rotations and radii as diverse starts
    starts = [base]
    for ph in (0.15, 0.3, 0.45, 0.6, 0.75):
        starts.append(_regular_polygon(N, cx=0.5, cy=0.5, r=0.5, phase=ph))
    for r in (0.45, 0.55):
        starts.append(_regular_polygon(N, cx=0.5, cy=0.5, r=r, phase=0.0))

    # Powell on smooth surrogate for each start
    for s in starts:
        if time.time() > deadline - 5.0:
            break
        x0 = s.reshape(-1)
        try:
            res = minimize(_smooth_obj, x0, args=(30.0,), method='Powell',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        except Exception:
            continue
        cand = res.x.reshape(N, 2)
        sc = _score(cand)
        if sc > best_score:
            best = cand
            best_score = sc

    # Exact local polish on the best
    if time.time() < deadline - 2.0:
        polished = _local_polish(best, deadline)
        sc = _score(polished)
        if sc > best_score:
            best = polished
            best_score = sc

    # Normalize: scale so convex hull area = 1 (unit area), centered
    h = _convex_hull_area(best)
    if h > 1e-12:
        scale = 1.0 / np.sqrt(h)
        c = best.mean(axis=0)
        best = (best - c) * scale
        # Re-center to keep coordinates reasonable
        best = best + 0.5

    return best

# EVOLVE-BLOCK-END
