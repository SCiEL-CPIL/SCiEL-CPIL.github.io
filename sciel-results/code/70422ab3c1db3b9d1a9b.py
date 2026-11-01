"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import cKDTree


def _score(c):
    """Largest common radius for centres c (N,3) in [-1/2,1/2]^3."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return 0.0
    tree = cKDTree(c)
    d = tree.query(c, k=2, workers=-1)[0][:, 1]
    nn = float(d.min())
    wall = float((0.5 - np.abs(c)).min())
    return min(nn * 0.5, wall)


def _local_opt(c, iters, step, deadline):
    """Local improvement; never decreases the score (monotone)."""
    c = c.astype(np.float64).copy()
    tree = cKDTree(c)
    best = _score(c)
    for _ in range(iters):
        if time.time() >= deadline:
            break
        d, idx = tree.query(c, k=2, workers=-1)
        nd = d[:, 1]
        ii = idx[:, 1]
        nd = np.maximum(nd, 1e-12)
        diff = c - c[ii]
        dirn = diff / nd[:, None]
        # Wall push (away from the nearest face).
        m = np.abs(c)
        k = np.argmin(m, axis=1)
        sgn = np.sign(c[np.arange(N), k])
        sgn[sgn == 0] = 1.0
        wall_dir = np.zeros_like(c)
        wall_dir[np.arange(N), k] = sgn
        # Step capped so the minimum constraint cannot decrease.
        h = step * np.minimum(nd, 0.5 - m.min(axis=1))
        h = np.maximum(h, 1e-12)
        c = c + h[:, None] * (dirn + wall_dir)
        c = np.clip(c, -0.5, 0.5)
        tree = cKDTree(c)
        s = _score(c)
        if s > best + 1e-15:
            best = s
    return c, best


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before the 150 s kill
    rng = np.random.default_rng(12345)

    # Seed: densest grid packing of 100 points (4x4x5 = 80, plus 20 extra).
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks], dtype=np.float64)
    base = grid[:N].copy()

    # Quick local polish of the plain grid (cheap, good baseline).
    base, best = _local_opt(base, 400, 0.5, deadline)
    best_c = base.copy()

    # Deterministic multi-restart local optimization.
    iters = 500
    step = 0.5
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() >= deadline - 3.0:
            break
        noise = rng.normal(0.0, 0.02, size=(N, 3))
        c = np.clip(base + noise, -0.5, 0.5)
        c, s = _local_opt(c, iters, step, deadline)
        if s > best + 1e-15:
            best = s
            best_c = c.copy()

    # Final safety: ensure a valid, finite result.
    if best <= 0.0:
        best_c = base
        best = _score(best_c)
    return best_c
# EVOLVE-BLOCK-END
