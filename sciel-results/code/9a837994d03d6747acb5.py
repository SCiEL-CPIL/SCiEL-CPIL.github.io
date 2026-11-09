"""Packing of 50 equal circles in a unit disk: multi-start simulated annealing + local polish."""
import numpy as np
import time

N = 50
_T0 = time.time()
_DEADLINE = _T0 + 110.0  # safety margin before the 120s kill (150s wall-clock budget)
_RNG = np.random.default_rng(20240517)


def _score(pts):
    """Largest common radius for the given centres (double precision)."""
    pts = np.asarray(pts, dtype=float)
    d = np.hypot(pts[:, 0], pts[:, 1])
    r = 1.0 - d.max()
    if N > 1:
        diff = pts[:, None, :] - pts[None, :, :]
        dmin = np.sqrt((diff * diff).sum(-1)).min()
        r = min(r, 0.5 * dmin)
    return r


def _soft_min(pts, beta):
    """Smooth surrogate: soft-min of (pairwise half-distances, container margin)."""
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(d, np.inf)
    pm = 0.5 * d
    cm = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    vals = np.concatenate([pm.ravel(), cm.ravel()])
    m = vals.min()
    return m + np.logaddexp.reduce(np.log(1.0 + np.exp(beta * (vals - m)))) / beta


def _local_polish(pts, beta, iters, step):
    """Gradient ascent on the soft-min objective with tiny steps."""
    pts = pts.copy()
    best = _score(pts)
    for _ in range(iters):
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt((diff * diff).sum(-1))
        np.fill_diagonal(d, np.inf)
        w = np.exp(-beta * d)
        w.fill_diagonal(0.0)
        grad = 2.0 * (w @ (pts - pts[None, :, :]))
        grad += beta * (1.0 - np.hypot(pts[:, 0], pts[:, 1])) / (np.hypot(pts[:, 0], pts[:, 1]) + 1e-12) * pts
        pts = pts + step * grad
        s = _score(pts)
        if s > best:
            best = s
    return pts, best


def _anneal(pts0, t_start, t_end, n_iter):
    """Simulated annealing on the soft-min surrogate; returns best valid config."""
    pts = pts0.copy()
    best = _score(pts)
    best_pts = pts.copy()
    s0 = _soft_min(pts, 10.0)
    n = max(1, n_iter)
    for it in range(n):
        if time.time() > _DEADLINE:
            break
        frac = it / n
        t = t_start * (t_end / t_start) ** frac
        cand = pts + _RNG.normal(0.0, t, size=pts.shape)
        s1 = _soft_min(cand, 10.0)
        if s1 >= s0 or _RNG.random() < np.exp((s1 - s0) / max(t, 1e-12)):
            pts = cand
            s0 = s1
            sc = _score(pts)
            if sc > best:
                best = sc
                best_pts = pts.copy()
    return best_pts, best


def _rings_init():
    """Concentric-ring configuration (known to be a strong local optimum for N=50)."""
    pts = [np.array([0.0, 0.0])]
    for r, k in [(0.38, 8), (0.74, 12), (1.0, 30)]:
        ang = _RNG.uniform(0.0, 2.0 * np.pi) + np.arange(k) * (2.0 * np.pi / k)
        for a in ang:
            pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
    return np.array(pts[:N])


def _hex_init():
    """Jittered hexagonal lattice clipped to the disk."""
    spacing = 0.21
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    pts = np.array(rows)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[rad <= 1.0 - spacing / 2.0]
    pts = pts + _RNG.normal(0.0, 0.002, size=pts.shape)
    order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
    return pts[order][:N]


def _random_init():
    """Random points in the disk."""
    r = np.sqrt(_RNG.uniform(0.0, 1.0, N))
    th = _RNG.uniform(0.0, 2.0 * np.pi, N)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 110.0
    best_pts = _rings_init()
    best = _score(best_pts)

    # Phase 1: multi-start annealing (rings, jittered hex, several random starts)
    starts = [best_pts.copy(), _hex_init()]
    for _ in range(3):
        starts.append(_random_init())
    for p0 in starts:
        if time.time() > _DEADLINE - 20.0:
            break
        p, s = _anneal(p0, t_start=0.06, t_end=0.0015, n_iter=9000)
        if s > best:
            best = s
            best_pts = p

    # Phase 2: high-precision local polish on the best configuration
    if time.time() < _DEADLINE - 5.0:
        best_pts, s = _local_polish(best_pts, beta=12.0, iters=1200, step=0.0004)
        if s > best:
            best = s
        # a second, tighter pass
        if time.time() < _DEADLINE - 2.0:
            best_pts, s = _local_polish(best_pts, beta=16.0, iters=800, step=0.00015)
            if s > best:
                best = s

    return np.asarray(best_pts, dtype=float)
