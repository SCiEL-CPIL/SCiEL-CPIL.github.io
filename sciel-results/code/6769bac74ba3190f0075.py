"""
Improved packing program using Voronoi-cell inradius optimization.

Approach:
- Represent the packing as a set of circle centers.
- The objective is to maximize the minimum "inradius" which is the minimum of:
    * half the distance to the nearest other center (non-overlap),
    * distance to the nearest wall (containment).
- We use a smooth soft-min objective over the active constraints (derived from
  Delaunay edges + walls) and optimize with L-BFGS-B.
- Multiple diverse initial configurations are tried.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 50


def _true_radius(centers):
    """Exact radius for a set of centers."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    pair = d.min() / 2.0
    wall = (0.5 - np.abs(centers)).min()
    return min(pair, wall)


def _delaunay_edges(centers):
    try:
        tri = Delaunay(centers)
        edges = set()
        for s in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = s[i], s[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _soft_min(values, k):
    """Differentiable approximation of min(values)."""
    m = values.min()
    s = values - m
    # Avoid overflow
    s = np.clip(s, 0, 50.0 / k)
    e = np.exp(-k * s)
    return m - np.log(e.sum()) / k


def _objective(params, k):
    c = params.reshape(N, 2)
    c = np.clip(c, -0.5, 0.5)
    edges = _delaunay_edges(c)
    vals = []
    for i, j in edges:
        vals.append(np.linalg.norm(c[i] - c[j]) / 2.0)
    vals.extend((0.5 - np.abs(c)).flatten())
    vals = np.array(vals)
    if vals.size == 0:
        return 0.0
    return -_soft_min(vals, k)


def _optimize(start, bounds, time_budget=25.0):
    best = start.copy()
    best_r = _true_radius(best)
    t0 = time.time()

    # Coarse pass
    try:
        res = minimize(
            _objective, best.ravel(), args=(300.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False}
        )
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
        r = _true_radius(c)
        if r > best_r:
            best_r, best = r, c
    except Exception:
        pass

    if time.time() - t0 > time_budget:
        return best, best_r

    # Fine pass
    try:
        res = minimize(
            _objective, best.ravel(), args=(2000.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False}
        )
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
        r = _true_radius(c)
        if r > best_r:
            best_r, best = r, c
    except Exception:
        pass

    return best, best_r


def _hex_init(seed, spacing=0.14):
    pts = []
    rng = np.random.RandomState(seed)
    rx = np.arange(-0.8, 0.8, spacing)
    ry = np.arange(-0.8, 0.8, spacing)
    for i, x in enumerate(rx):
        off = spacing / 2 if i % 2 == 0 else 0.0
        for y in ry:
            pts.append([x + off, y])
    pts = np.array(pts)
    mask = (np.abs(pts[:, 0]) <= 0.49) & (np.abs(pts[:, 1]) <= 0.49)
    pts = pts[mask]
    if len(pts) < N:
        return None
    idx = rng.choice(len(pts), N, replace=False)
    return pts[idx]


def _square_init(seed):
    gx = np.linspace(-0.45, 0.45, 7)
    gy = np.linspace(-0.45, 0.45, 7)
    pts = np.array([[x, y] for x in gx for y in gy])
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pts), N, replace=False)
    return pts[idx]


def _random_init(seed, lo=-0.45, hi=0.45):
    rng = np.random.RandomState(seed)
    return rng.uniform(lo, hi, (N, 2))


def _jittered_hex_init(seed, jitter=0.02):
    base = _hex_init(seed)
    if base is None:
        return None
    rng = np.random.RandomState(seed + 1)
    return np.clip(base + rng.normal(0, jitter, (N, 2)), -0.49, 0.49)


def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)

    inits = []
    h = _hex_init(42)
    if h is not None:
        inits.append(h)
    inits.append(_square_init(123))
    inits.append(_random_init(999))
    jh = _jittered_hex_init(43)
    if jh is not None:
        inits.append(jh)
    inits.append(_random_init(1000))
    # Randomized grid
    rng = np.random.RandomState(555)
    gx = np.sort(rng.uniform(-0.49, 0.49, 7))
    gy = np.sort(rng.uniform(-0.49, 0.49, 7))
    pts = np.array([[x, y] for x in gx for y in gy])
    idx = rng.choice(len(pts), N, replace=False)
    inits.append(pts[idx])

    best_centers = None
    best_score = -1.0
    t_start = time.time()

    for init in inits:
        if time.time() - t_start > 100.0:
            break
        centers, score = _optimize(init, bounds)
        if score > best_score:
            best_score = score
            best_centers = centers

    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))

    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    return best_centers
