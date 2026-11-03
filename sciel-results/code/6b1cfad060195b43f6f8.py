"""Fifty equal spheres in a unit cube - maximize common radius.

Design:
  * Objective  r = min( min_{i<j} d_ij/2 , min_i min_k (0.5 - |c_ik|) )
  * Nonsmooth min is handled two ways:
      - soft-min (log-sum-exp) surrogate -> smooth gradient ascent
      - pairwise inverse-square repulsion -> robust force for close pairs
  * Deterministic (cubic-grid) starts + random restarts under a wall-clock budget.
  * Final polish with tighter tolerances.
"""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0          # wall-clock safety margin (evaluator kills at 150 s)
HALF = 0.5                # cube is [-0.5, 0.5]^3


def _score(pts):
    """Exact radius for a valid configuration (0 if invalid)."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 3) or not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -HALF - 1e-9) or np.any(pts > HALF + 1e-9):
        return 0.0
    dmin = cKDTree(pts).query(pts, k=2)[0][:, 1].min()
    wall = (HALF - np.abs(pts)).min()
    return min(0.5 * dmin, wall)


def _softmin_neglogsumexp(v, beta):
    """-logsumexp(v/beta)*beta  (smooth approximation of min(v)) and its grad."""
    vb = v / beta
    m = vb.max()
    z = np.exp(vb - m)
    s = z.sum()
    val = -beta * (m + np.log(s))
    grad = -z / s * beta
    return val, grad


def _optimize(pts, t0, time_limit):
    """Local refinement of a starting configuration. Returns (best, best_pts)."""
    pts = np.array(pts, dtype=float, copy=True)
    best = _score(pts)
    best_pts = pts.copy()

    def wall_ok():
        return (time.time() - t0) < time_limit

    # ---- Phase A: soft-min gradient ascent (smooth, global-ish) -------------
    beta = 0.02
    lr = 1e-3
    for it in range(3000):
        if not wall_ok():
            break
        d = cKDTree(pts).query(pts, k=2)[0][:, 1]
        d = np.maximum(d, 1e-6)
        wall = HALF - np.abs(pts)
        v = np.concatenate([d / 2.0, wall.ravel()])
        val, grad = _softmin_neglogsumexp(v, beta)
        g = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                g[i] += grad[j] * (pts[i] - pts[j]) / d[i]
        for i in range(N):
            for k in range(3):
                g[i, k] += grad[N + 3 * i + k] * (-np.sign(pts[i, k]))
        pts = np.clip(pts + lr * g, -HALF, HALF)
        s = _score(pts)
        if s > best:
            best = s
            best_pts = pts.copy()

    # ---- Phase B: pairwise inverse-square repulsion (robust near minima) ----
    for it in range(20000):
        if not wall_ok():
            break
        F = np.zeros_like(pts)
        for i in range(N):
            pi = pts[i]
            for j in range(N):
                if i == j:
                    continue
                diff = pi - pts[j]
                d2 = float(diff @ diff)
                if d2 < 1e-12:
                    diff = diff + np.random.normal(0, 1e-4, 3)
                    d2 = float(diff @ diff)
                d = np.sqrt(d2)
                F[i] += diff / (d * d2)
        for i in range(N):
            for k in range(3):
                if pts[i, k] < -HALF:
                    F[i, k] += 10.0
                elif pts[i, k] > HALF:
                    F[i, k] -= 10.0
        step = 1e-4
        pts = np.clip(pts + step * F, -HALF, HALF)
        if it % 200 == 0:
            s = _score(pts)
            if s > best:
                best = s
                best_pts = pts.copy()
    s = _score(pts)
    if s > best:
        best = s
        best_pts = pts.copy()
    return best, best_pts


def _grid_start():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    t0 = time.time()
    best = 0.0
    best_pts = None
    rng = np.random.default_rng(12345)

    # Deterministic cubic-grid start first (guaranteed-valid baseline).
    g = _grid_start()
    b, bp = _optimize(g, t0, DEADLINE)
    if b > best:
        best, best_pts = b, bp

    # Random restarts while time remains (each starts from a fresh jittered grid).
    while (time.time() - t0) < (DEADLINE - 6.0):
        start = g + rng.normal(0.0, 0.03, g.shape)
        start = np.clip(start, -HALF, HALF)
        b, bp = _optimize(start, t0, DEADLINE)
        if b > best:
            best, best_pts = b, bp

    if best_pts is None:
        return g
    return best_pts
