"""Heilbronn convex regions: maximize minimum triangle area over convex hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all C(N,3) index triples
_TRI_IDX = np.array(list(combinations(range(N), 3)), dtype=np.int64)

def _all_areas(pts):
    """Vectorized computation of all C(n,3) triangle areas. pts: (n,2)."""
    a = pts[_TRI_IDX[:, 0]]
    b = pts[_TRI_IDX[:, 1]]
    c = pts[_TRI_IDX[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))

def _hull_area(pts):
    """Area of the convex hull of pts."""
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    areas = _all_areas(pts)
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return float(np.min(areas) / ha)

def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)

def _soft_min_objective(pts, T):
    """Smooth surrogate: T * logsumexp(-normalized_areas/T). Minimize to maximize min area."""
    areas = _all_areas(pts)
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 1e6
    x = areas / ha
    m = np.max(x)
    return T * (np.log(np.sum(np.exp(-(x - m) / T))) + (-(x - m) / T).max() + m / T)

# ---------- Initializers (diverse, structured) ----------
def _regular_polygon(n):
    a = 2 * np.pi * np.arange(n) / n
    p = np.column_stack([np.cos(a), np.sin(a)])
    return p / np.sqrt(_hull_area(p))

def _halton(n, base=(2, 3)):
    def halton_seq(i, b):
        f, r = 1.0, 0.0
        while i > 0:
            f /= b
            r += f * (i % b)
            i //= b
        return r
    pts = np.array([[halton_seq(i + 1, base[0]), halton_seq(i + 1, base[1])] for i in range(n)])
    c = pts.mean(axis=0)
    pts = pts - c
    s = np.max(np.abs(pts))
    if s > 0:
        pts = pts / s
    return pts

def _concentric_rings(n):
    """One center + 6 + 6 = 13 points on two rings (hexagonal symmetry)."""
    pts = [np.array([0.0, 0.0])]
    r1 = 0.5
    for k in range(6):
        a = 2 * np.pi * k / 6
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    r2 = 1.0
    for k in range(6):
        a = 2 * np.pi * k / 6 + np.pi / 6
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    p = np.array(pts)
    return p / np.sqrt(_hull_area(p))

def _jittered_grid(n, rng):
    """Jittered 4x4 grid (drop one to get 13)."""
    g = np.array([[i, j] for j in range(4) for i in range(4)], dtype=float)
    g = g / 3.0
    g = g - 0.5
    g = g + rng.normal(0, 0.05, g.shape)
    p = g[:n]
    return p / np.sqrt(_hull_area(p))

def _fibonacci(n):
    """Fibonacci (spherical) disk packing projected to 2D."""
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    r = np.sqrt((i + 0.5) / n)
    theta = 2 * np.pi * i / golden
    p = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return p / np.sqrt(_hull_area(p))

def _random_pts(n, rng):
    p = rng.uniform(-1, 1, size=(n, 2))
    return p / np.sqrt(_hull_area(p))

# ---------- Optimization phases ----------
def _anneal_soft_min(pts, T0, T_end, n_steps, deadline):
    """Anneal the log-sum-exp temperature, running L-BFGS-B at each level."""
    x = pts.flatten()
    T = T0
    for s in range(n_steps):
        if time.time() > deadline - 3.0:
            break
        T = T0 * (T_end / T0) ** (s / max(1, n_steps - 1))
        try:
            res = minimize(
                _soft_min_objective, x, args=(T,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            if res.x is not None and np.all(np.abs(res.x) < 50):
                x = res.x
        except Exception:
            pass
    return x.reshape(N, 2)

def _nelder_mead(pts, deadline):
    """Refine with Nelder-Mead on the true objective."""
    try:
        res = minimize(
            _neg_score, pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if res.x is not None and np.all(np.abs(res.x) < 50):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return pts

def _perturb_restart(pts, rng, deadline):
    """Local perturbation + quick NM to escape local minima."""
    x = pts.flatten()
    best = pts
    bs = _score(pts)
    for _ in range(8):
        if time.time() > deadline - 3.0:
            break
        scale = 0.03 * (1 + rng.random())
        x2 = x + rng.normal(0, scale, x.shape)
        p2 = x2.reshape(N, 2)
        p2 = _nelder_mead(p2, deadline)
        s2 = _score(p2)
        if s2 > bs:
            best, bs = p2, s2
            x = p2.flatten()
    return best, bs

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(20240607)

    best_pts = None
    best_score = -1.0

    def consider(pts):
        nonlocal best_pts, best_score
        if pts is None:
            return
        if not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Build diverse initial pool
    pool = [
        _regular_polygon(N),
        _halton(N, base=(2, 3)),
        _halton(N, base=(3, 5)),
        _concentric_rings(N),
        _fibonacci(N),
        _jittered_grid(N, rng),
        _random_pts(N, rng),
        _random_pts(N, rng),
    ]

    # Phase 1: annealed soft-min on each diverse start
    for p0 in pool:
        if time.time() > deadline - 10.0:
            break
        p = _anneal_soft_min(p0, T0=0.30, T_end=0.005, n_steps=6, deadline=deadline)
        consider(p)

    # Phase 2: Nelder-Mead refinement on current best
    if best_pts is not None and time.time() < deadline - 15.0:
        p = _nelder_mead(best_pts, deadline)
        consider(p)

    # Phase 3: perturbation restarts from best (escape local minima)
    if best_pts is not None and time.time() < deadline - 10.0:
        p, s = _perturb_restart(best_pts, rng, deadline)
        consider(p)

    # Phase 4: a couple more fresh diverse starts + anneal if time remains
    if time.time() < deadline - 15.0:
        for p0 in [_random_pts(N, rng), _concentric_rings(N), _halton(N, base=(5, 7))]:
            if time.time() > deadline - 10.0:
                break
            p = _anneal_soft_min(p0, T0=0.25, T_end=0.005, n_steps=4, deadline=deadline)
            consider(p)
            if time.time() < deadline - 8.0:
                consider(_nelder_mead(p, deadline))

    # Phase 5: final NM polish on the global best
    if best_pts is not None and time.time() < deadline - 4.0:
        consider(_nelder_mead(best_pts, deadline))

    if best_pts is None:
        return _regular_polygon(N)
    return best_pts
# EVOLVE-BLOCK-END
