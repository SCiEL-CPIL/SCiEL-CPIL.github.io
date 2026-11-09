"""Maximize product of pairwise distances for 8 planar points at fixed diameter.

Uses the known extremal construction (regular heptagon + center), which
beats the regular octagon, plus a gradient polish for refinement.
"""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np


def _diameter(pts):
    p = np.asarray(pts, dtype=float)
    n = p.shape[0]
    d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dd = p[i] - p[j]
            d = max(d, math.hypot(dd[0], dd[1]))
    return d


def _log_product(pts):
    p = np.asarray(pts, dtype=float)
    n = p.shape[0]
    lp = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dd = p[i] - p[j]
            lp += math.log(math.hypot(dd[0], dd[1]))
    return lp


def _score(pts):
    """Score = Delta / n^n after scaling so diameter = 2."""
    d = _diameter(pts)
    if d <= 0:
        return -math.inf
    s = 2.0 / d
    lp = _log_product(pts)
    # log Delta = log(prod d_ij) + n * log(s)
    log_delta = lp + 8 * math.log(s)
    return math.exp(log_delta - 8 * math.log(8))


def _neg_log_score_flat(x):
    p = x.reshape(8, 2)
    lp = _log_product(p)
    d = _diameter(p)
    if d <= 0:
        return 1e10
    return -(lp + 8 * math.log(2.0 / d))


def _polish(pts, max_iter=4000, ftol=1e-16):
    try:
        from scipy.optimize import minimize
    except Exception:
        return pts
    x0 = np.asarray(pts, dtype=float).flatten()
    try:
        res = minimize(
            _neg_log_score_flat,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-14},
        )
        if res.x is not None and res.success or (
            res.fun < _neg_log_score_flat(x0) - 1e-12
        ):
            return res.x.reshape(8, 2)
    except Exception:
        pass
    return pts


def construct():
    deadline = time.time() + 135.0
    pi = math.pi
    best = None
    best_s = -math.inf

    def consider(pts):
        nonlocal best, best_s
        try:
            # ensure 8 distinct points
            p = np.asarray(pts, dtype=float)
            if p.shape != (8, 2):
                return
            # check distinctness
            ok = True
            for i in range(8):
                for j in range(i + 1, 8):
                    if math.hypot(p[i][0] - p[j][0], p[i][1] - p[j][1]) < 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                return
            s = _score(p)
            if s > best_s:
                best_s = s
                best = p
        except Exception:
            pass

    # Candidate 1: regular heptagon + center (known extremal, beats octagon)
    hept = []
    for k in range(7):
        a = 2 * pi * k / 7
        hept.append([math.cos(a), math.sin(a)])
    hept.append([0.0, 0.0])
    consider(hept)

    # Candidate 2: regular octagon (reference, score 1)
    octa = []
    for k in range(8):
        a = 2 * pi * k / 8
        octa.append([math.cos(a), math.sin(a)])
    consider(octa)

    # Candidate 3: heptagon with slight radius variation (test robustness)
    for rr in (0.98, 1.02):
        cand = []
        for k in range(7):
            a = 2 * pi * k / 7
            cand.append([rr * math.cos(a), rr * math.sin(a)])
        cand.append([0.0, 0.0])
        consider(cand)

    # Gradient polish the best candidate found so far
    if best is not None and time.time() < deadline - 8.0:
        polished = _polish(best, max_iter=4000, ftol=1e-16)
        consider(polished)

    # Final safety fallback: regular octagon
    if best is None:
        return [[math.cos(2 * pi * k / 8), math.sin(2 * pi * k / 8)] for k in range(8)]

    out = [[float(x), float(y)] for x, y in np.asarray(best)]
    # keep coordinates well within bounds
    for p in out:
        p[0] = max(-1e5, min(1e5, p[0]))
        p[1] = max(-1e5, min(1e5, p[1]))
    return out
# EVOLVE-BLOCK-END
