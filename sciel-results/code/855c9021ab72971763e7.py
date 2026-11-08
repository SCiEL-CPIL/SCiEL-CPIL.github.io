"""
Heilbronn n=14: maximize (min triangle area) / (convex hull area).
Fast, multi-start BFGS on a smooth soft-min objective + Nelder-Mead polish.
"""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 138.0  # safety margin before the 150 s kill


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def _all_areas(pts):
    """All C(N,3) triangle areas, vectorized, flat array."""
    n = pts.shape[0]
    x = pts[:, 0]
    y = pts[:, 1]
    out = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            out.append(0.5 * np.abs((x[j] - x[i]) * (y[j + 1:] - y[i])
                                    - (x[j + 1:] - x[i]) * (y[j] - y[i])))
    return np.concatenate(out)


def _score(pts):
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    a = _all_areas(pts)
    return float(np.min(a) / hull_a)


def _soft_obj(flat, beta):
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e12
    a = _all_areas(pts)
    if a.size == 0 or np.min(a) < 1e-14:
        return 1e12
    z = -beta * a
    mz = np.max(z)
    if mz < -500:
        lse = mz
    else:
        lse = mz + np.log(np.sum(np.exp(z - mz)))
    soft_min = -lse / beta
    if soft_min < 1e-14:
        return 1e12
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e12
    return ha / soft_min


def _neg_score(flat):
    return -_score(flat.reshape(N, 2))


def _make_starts():
    starts = []
    # Regular polygon
    ang = 2 * np.pi * np.arange(N) / N
    starts.append(np.column_stack([np.cos(ang), np.sin(ang)]))
    # Ring: 7 outer + 7 inner
    a7 = 2 * np.pi * np.arange(7) / 7
    a7b = a7 + np.pi / 7
    starts.append(np.vstack([
        np.column_stack([np.cos(a7), np.sin(a7)]),
        np.column_stack([0.5 * np.cos(a7b), 0.5 * np.sin(a7b)]),
    ]))
    # Hexagonal-ish lattice in a disk
    rng = np.random.RandomState(7)
    pts = []
    for r in [0.0, 0.5, 1.0]:
        if r == 0.0:
            pts.append([0.0, 0.0])
        else:
            m = int(round(6 * r))
            aa = 2 * np.pi * np.arange(m) / m + (0.3 if r > 0.5 else 0.0)
            pts.extend((r * np.cos(aa), r * np.sin(aa)))
    starts.append(np.array(pts[:N]))
    # Random starts
    for s in range(6):
        rr = rng.rand(N) ** 0.5
        th = 2 * np.pi * rng.rand(N)
        starts.append(np.column_stack([rr * np.cos(th), rr * np.sin(th)]))
    return starts


def construct():
    t0 = time.time()

    def rem():
        return DEADLINE - (time.time() - t0)

    best_pts = None
    best_score = 0.0

    for st in _make_starts():
        if rem() < 12:
            break
        s = _score(st)
        if s > best_score:
            best_score = s
            best_pts = st.copy()

    # Multi-start BFGS with beta annealing
    for st in _make_starts():
        if rem() < 12:
            break
        x0 = st.flatten()
        for beta in (25.0, 60.0, 150.0):
            if rem() < 12:
                break
            try:
                r = minimize(lambda x: _soft_obj(x, beta), x0, method='BFGS',
                             options={'maxiter': 400, 'gtol': 1e-9})
                x0 = r.x
            except Exception:
                pass
        cand = x0.reshape(N, 2)
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand.copy()

    # Final Nelder-Mead polish on the hard objective
    if best_pts is not None and rem() > 8:
        try:
            r = minimize(_neg_score, best_pts.flatten(), method='Nelder-Mead',
                         options={'maxiter': 20000, 'xatol': 1e-12,
                                  'fatol': 1e-14, 'adaptive': True})
            cand = r.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
        except Exception:
            pass

    if best_pts is None:
        ang = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(ang), np.sin(ang)])

    return best_pts
