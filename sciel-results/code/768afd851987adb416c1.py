import time
import numpy as np
from scipy.optimize import minimize

N = 11
H = np.sqrt(3.0) / 2.0
PERIM = 3.0

_T0 = None


def _arc_to_point(s):
    s = s % PERIM
    if s < 1.0:
        return np.array([s, 0.0])
    elif s < 2.0:
        t = s - 1.0
        return np.array([1.0 - 0.5 * t, H * t])
    else:
        t = s - 2.0
        return np.array([0.5 * (3.0 - s), H * (3.0 - s)])


def _points_from_params(p):
    # p: (N+1)-dim. first N -> softmax gaps; last -> rotation offset
    z = p[:N] - p[:N].max()
    w = np.exp(z)
    gaps = w / w.sum() * PERIM
    offset = PERIM * (0.5 + 0.5 * np.tanh(p[N]))
    verts = []
    acc = offset
    for i in range(N):
        verts.append(acc)
        acc += gaps[i]
    return np.array([_arc_to_point(s) for s in verts])


def _min_tri_area(pts):
    n = len(pts)
    best = np.inf
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            for k in range(j + 1, n):
                pk = pts[k]
                area = 0.5 * abs(np.cross(pj - pi, pk - pi))
                if area < best:
                    best = area
    return best


def _objective(p):
    return -_min_tri_area(_points_from_params(p))


def construct():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    deadline = _T0 + 130.0

    rng = np.random.default_rng(2024)
    best_pts = None
    best_val = -np.inf

    def consider(p):
        nonlocal best_pts, best_val
        pts = _points_from_params(p)
        v = _min_tri_area(pts)
        if v > best_val:
            best_val = v
            best_pts = pts.copy()

    n_restarts = 80
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        if r == 0:
            p0 = np.zeros(N + 1)
        else:
            p0 = rng.normal(0.0, 1.0, size=N + 1)
        try:
            res = minimize(_objective, p0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            consider(res.x)
        except Exception:
            pass

    if best_pts is None:
        verts = [i * PERIM / N for i in range(N)]
        best_pts = np.array([_arc_to_point(s) for s in verts])

    return np.asarray(best_pts, dtype=float)
