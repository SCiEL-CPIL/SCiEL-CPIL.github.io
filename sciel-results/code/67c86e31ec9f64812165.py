"""Sphere packing in a unit cube: repulsive-force local search + polish."""
import time
import numpy as np

N = 50
_DEADLINE = 150.0 - 12.0          # safety margin before the 150 s kill
_T = 30.0                          # temperature scale for the repulsive kernel
_MAX_STEPS = 150000


def _score(pts):
    """Largest common radius (higher = better)."""
    n = pts.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        d[i, i + 1:] = np.linalg.norm(pts[i] - pts[i + 1:], axis=1)
    d = np.maximum(d, d.T)
    r = 0.5 * np.min(d[np.triu_indices(n, k=1)])
    r = min(r, 0.5 - np.max(np.abs(pts)))
    return r


def _repulsive(pts, step):
    """One vectorized repulsive-force step (in-place). Returns new step size."""
    n = pts.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        d[i, i + 1:] = np.linalg.norm(pts[i] - pts[i + 1:], axis=1)
    d = np.maximum(d, d.T)
    np.fill_diagonal(d, 1.0)
    diff = pts[:, None, :] - pts[None, :, :]          # (n,n,3): c_i - c_j
    f = np.exp(-_T / d) / (d * d)[:, :, None]          # repulsion strength
    force = np.sum(f * diff, axis=1)                    # (n,3)
    wall = np.zeros((n, 3))
    for k in range(3):
        x = pts[:, k]
        wall[:, k] += np.where(x > 0.0, -1.0, 1.0) * np.exp(-_T * (0.5 - np.abs(x)))
    pts += step * (force + wall)
    return step * 0.99985


def _polish(pts):
    """High-precision polish: tiny steps, force only (no wall push), keep best."""
    best = pts.copy()
    best_r = _score(pts)
    step = 1e-5
    n = pts.shape[0]
    for it in range(3000):
        d = np.zeros((n, n))
        for i in range(n):
            d[i, i + 1:] = np.linalg.norm(pts[i] - pts[i + 1:], axis=1)
        d = np.maximum(d, d.T)
        np.fill_diagonal(d, 1.0)
        diff = pts[:, None, :] - pts[None, :, :]
        f = np.exp(-_T / d) / (d * d)[:, :, None]
        force = np.sum(f * diff, axis=1)
        pts = pts + step * force
        r = _score(pts)
        if r > best_r:
            best_r = r
            best = pts.copy()
        step *= 0.999
    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best = None
    best_r = -1.0

    def try_start(pts, steps):
        nonlocal best, best_r
        pts = pts.copy()
        step = 0.02
        for it in range(steps):
            if time.time() > _DEADLINE:
                break
            pts = _repulsive(pts, step)
            if it % 1500 == 0:
                r = _score(pts)
                if r > best_r:
                    best_r = r
                    best = pts.copy()
        r = _score(pts)
        if r > best_r:
            best_r = r
            best = pts.copy()

    # 1) random starts
    for restart in range(8):
        if time.time() > _DEADLINE:
            break
        pts = (rng.random((N, 3)) - 0.5) * 0.98
        try_start(pts, _MAX_STEPS)

    # 2) jittered grid starts (grid is a decent, dense local basin)
    if best is not None and time.time() < _DEADLINE:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N].copy()
        for restart in range(4):
            if time.time() > _DEADLINE:
                break
            pts = grid + rng.normal(0, 0.05, (N, 3))
            pts = np.clip(pts, -0.49, 0.49)
            try_start(pts, _MAX_STEPS)

    # 3) final high-precision polish on the best packing
    if best is not None and time.time() < _DEADLINE:
        polished, pr = _polish(best)
        if pr > best_r:
            best, best_r = polished, pr

    if best is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]

    # guarantee validity: strictly inside the cube
    best = np.clip(best, -0.4999999, 0.4999999)
    return best
