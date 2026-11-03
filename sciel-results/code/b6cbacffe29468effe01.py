import time
import numpy as np
from scipy.spatial import cKDTree

N = 50
TIME_LIMIT = 140.0          # evaluator kills at 150s wall-clock; keep a safety margin
RNG_SEED = 123456789

# --- objective (exact, double precision, matches the evaluator) --------------
def _score(c):
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return -1.0
    tree = cKDTree(c)
    d, _ = tree.query(c, k=2)
    r = float(np.min(d[:, 1]) * 0.5)
    m = np.min(np.abs(c), axis=1)
    r = min(r, float(0.5 - np.max(m)))
    return r


# --- deterministic local search ---------------------------------------------
def _local_search(c0, deadline, rng, iters_cap=400000):
    c = np.array(c0, dtype=float)
    best = _score(c)
    step = 0.02
    n = c.shape[0]
    it = 0
    while it < iters_cap:
        it += 1
        if time.time() > deadline:
            break
        i = int(rng.integers(0, n))
        k = int(rng.integers(0, 3))
        for sgn in (1, -1):
            cand = c.copy()
            cand[i, k] += sgn * step
            if cand[i, k] < -0.5 or cand[i, k] > 0.5:
                continue
            sc = _score(cand)
            if sc > best + 1e-12:
                c = cand
                best = sc
                step = min(step * 1.05, 0.05)
                break
        else:
            step *= 0.8
            if step < 1e-6:
                break
    return c, best


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT
    rng = np.random.default_rng(RNG_SEED)

    # good deterministic start: 4x4x4 grid, keep 50
    ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = pts[:N].copy()
    best_c = c.copy()
    best_r = _score(c)

    # 1) deterministic refinement
    c, r = _local_search(c, deadline, rng)
    if r > best_r:
        best_r, best_c = r, c
    if time.time() > deadline:
        return best_c

    # 2) a few random restarts (cheap, each guarded by the same deadline)
    for _ in range(6):
        if time.time() > deadline:
            break
        c0 = (rng.random((N, 3)) - 0.5) * 0.9
        c0 = np.clip(c0, -0.5 + 1e-6, 0.5 - 1e-6)
        c, r = _local_search(c0, deadline, rng, iters_cap=200000)
        if r > best_r:
            best_r, best_c = r, c

    # 3) final high-precision polish from the best configuration
    if time.time() < deadline:
        c, r = _local_search(best_c, deadline, rng, iters_cap=300000)
        if r > best_r:
            best_r, best_c = r, c

    return best_c
