"""Heilbronn convex regions n=14: maximize min triangle area / hull area.

Structural change: global search via Differential Evolution over the full
28D point space (escapes the local minima that trap Nelder-Mead/SLSQP),
with a fast vectorized objective and a short exact-objective polish.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14

# ---- precomputed triangle index arrays (constant) ----
_I, _J, _K = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _I.append(i); _J.append(j); _K.append(k)
_I = np.asarray(_I); _J = np.asarray(_J); _K = np.asarray(_K)
_M = len(_I)


def _triangle_areas(p):
    """All C(N,3) triangle areas, vectorized."""
    pi = p[_I]; pj = p[_J]; pk = p[_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(p):
    """Convex hull area of the point set."""
    try:
        return ConvexHull(p).volume
    except Exception:
        return 0.0


def _score(p):
    """min triangle area / convex hull area."""
    hull = _hull_area(p)
    if hull <= 1e-12:
        return 0.0
    return float(np.min(_triangle_areas(p)) / hull)


# ---- fast objective with a cached hull (DE moves points little per iter) ----
_HULL_CACHE = {}


def _fast_neg_score(x):
    p = x.reshape(-1, 2)
    key = p.tobytes()
    hull = _HULL_CACHE.get(key)
    if hull is None:
        hull = _hull_area(p)
        _HULL_CACHE[key] = hull
    if hull <= 1e-12:
        return 1e6
    min_tri = np.min(_triangle_areas(p))
    return -min_tri / hull


def _fast_neg_score_reset(x):
    _HULL_CACHE.clear()
    return _fast_neg_score(x)


def _regular_polygon(n, radius=1.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def _two_rings(n, inner_frac=0.5):
    n_outer = n // 2
    n_inner = n - n_outer
    oa = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    ia = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([np.cos(oa), np.sin(oa)])
    inner = inner_frac * np.column_stack([np.cos(ia), np.sin(ia)])
    return np.vstack([outer, inner])


def _three_rings(n):
    # 6 outer, 5 middle, 3 inner
    rings = [(6, 1.0, 0.0), (5, 0.62, np.pi / 5), (3, 0.30, np.pi / 3)]
    pts = []
    for cnt, r, off in rings:
        a = np.linspace(0, 2 * np.pi, cnt, endpoint=False) + off
        pts.append(r * np.column_stack([np.cos(a), np.sin(a)]))
    return np.vstack(pts)[:n]


def _grid_like(n):
    # points on a 4x4 grid (16), take 14
    g = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    return g[:n]


def _initial_candidates():
    cands = []
    cands.append(_regular_polygon(N))
    for f in (0.4, 0.5, 0.6, 0.7):
        cands.append(_two_rings(N, f))
    cands.append(_three_rings(N))
    cands.append(_grid_like(N))
    rng = np.random.RandomState(42)
    for _ in range(6):
        base = _regular_polygon(N)
        cands.append(base + rng.normal(0, 0.10, (N, 2)))
    return cands


def _polish(p, deadline, maxiter=1500):
    """Short exact-objective Nelder-Mead polish."""
    if time.time() > deadline - 3:
        return p, _score(p)
    best = p.copy()
    best_s = _score(p)
    try:
        res = minimize(
            lambda x: -_score(x.reshape(-1, 2)),
            p.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True},
        )
        cand = res.x.reshape(-1, 2)
        s = _score(cand)
        if s > best_s:
            best, best_s = cand, s
    except Exception:
        pass
    return best, best_s


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    # ---- seed with best analytic candidate ----
    cands = _initial_candidates()
    best_p = cands[0].copy()
    best_s = _score(best_p)
    for c in cands[1:]:
        s = _score(c)
        if s > best_s:
            best_s, best_p = s, c.copy()

    # ---- global search: Differential Evolution on the 28D space ----
    # box in [-1.5, 1.5]^28; normalization is invariant to scale/rotation,
    # so the wide box lets DE explore shape while scale is irrelevant.
    bounds = [(-1.5, 1.5)] * (2 * N)
    x0 = best_p.flatten()
    try:
        _HULL_CACHE.clear()
        de = differential_evolution(
            _fast_neg_score_reset,
            bounds,
            x0=x0,
            seed=1,
            popsize=16,
            maxiter=250,
            tol=1e-7,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            updating='immediate',
            callback=lambda xk, convergence: (
                _HULL_CACHE.clear() if convergence > 0.5 else None
            ),
            init='latinhypercube',
        )
        if de.success or de.x is not None:
            cand = de.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best_p = s, cand
    except Exception:
        pass

    # ---- second DE run from a different seed / popsize for diversity ----
    if time.time() < deadline - 25:
        try:
            _HULL_CACHE.clear()
            de2 = differential_evolution(
                _fast_neg_score_reset,
                bounds,
                x0=_two_rings(N, 0.5).flatten(),
                seed=7,
                popsize=20,
                maxiter=180,
                tol=1e-7,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                updating='immediate',
                init='sobol',
            )
            cand = de2.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best_p = s, cand
        except Exception:
            pass

    # ---- exact-objective polish ----
    if time.time() < deadline - 6:
        p, s = _polish(best_p, deadline, maxiter=2000)
        if s > best_s:
            best_s, best_p = s, p

    # ---- final micro-polish ----
    if time.time() < deadline - 3:
        p, s = _polish(best_p, deadline, maxiter=800)
        if s > best_s:
            best_s, best_p = s, p

    return best_p
