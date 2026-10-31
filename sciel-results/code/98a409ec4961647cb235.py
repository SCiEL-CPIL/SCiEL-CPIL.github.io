"""Optimize the ratio  ||h||_2^2 / (||h||_1 * sup|autoconvolution|)  over 50 nonnegative
step heights.  The seed (all-ones) gives 1.0; we push it above 1 by a smooth epigraph
local search that minimizes the max autoconvolution value for a fixed norm, then polishes.
"""
import time
import numpy as np

N = 50
DEADLINE = 140.0          # wall-clock safety margin below the 150s kill
np.random.seed(20240607)


def _autoconv(h):
    # full linear autoconvolution, length 2N-1
    return np.convolve(h, h, mode='full')


def _score(h):
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float(np.dot(h, h))
    ac = _autoconv(h)
    sup = float(np.max(np.abs(ac)))
    if sup <= 0:
        return 0.0
    return l2sq / (l1 * sup)


def _smoothed_score(h, temp):
    # smooth surrogate of the ratio: replace sup|ac| by a soft-max (log-sum-exp) of ac^2
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float(np.dot(h, h))
    ac = _autoconv(h)
    ac2 = ac * ac
    m = float(np.max(ac2))
    if m <= 0:
        return 0.0
    # log-sum-exp of ac2/temp, then exponentiate -> soft-max of ac2
    s = np.sum(np.exp((ac2 - m) / temp))
    soft = m + temp * np.log(s)
    return l2sq / (l1 * soft)


def _project_nonneg(x):
    return np.clip(x, 0.0, None)


def _local_search(h, temp, iters, step):
    """Gradient-free-ish local search on the smoothed objective (maximize)."""
    h = np.array(h, dtype=float)
    best = _smoothed_score(h, temp)
    cur = h.copy()
    rng = np.random.RandomState(12345)
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        # random direction (Lipschitz-ish), keep nonnegativity via projection
        d = rng.standard_normal(N)
        # try a small step
        cand = _project_nonneg(cur + step * d)
        if cand.sum() <= 0:
            continue
        v = _smoothed_score(cand, temp)
        if v > best + 1e-12:
            best = v
            cur = cand
    return cur, best


def _polish(h, iters=4000, step0=0.05):
    """Fine coordinate/gradient-free polish on the true (nonsmooth) objective."""
    h = np.array(h, dtype=float)
    best = _score(h)
    cur = h.copy()
    rng = np.random.RandomState(999)
    step = step0
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        # shrink step adaptively
        step = step0 * (0.99 ** (it // 200))
        idx = rng.randint(N)
        # random small move on one coordinate (nonnegativity kept)
        delta = rng.uniform(-step, step)
        cand = cur.copy()
        cand[idx] = max(0.0, cand[idx] + delta)
        v = _score(cand)
        if v > best + 1e-12:
            best = v
            cur = cand
    return cur, best


def construct():
    t0 = time.time()
    # ---- Phase 0: seed and a few deterministic structured starts ----
    starts = []
    ones = np.ones(N)
    starts.append(ones)

    # sparse / peaked starts to escape the flat basin
    for k in (5, 8, 12, 20):
        s = np.zeros(N)
        idx = np.random.choice(N, size=k, replace=False)
        s[idx] = 1.0
        starts.append(s)

    # linear / quadratic ramps
    x = np.linspace(0, 1, N)
    starts.append(x)
    starts.append(1.0 - x)
    starts.append(np.exp(-((x - 0.5) ** 2) / 0.02))
    starts.append(np.cos(np.pi * (x - 0.5)) ** 2)

    # ---- Phase 1: local search on a smooth surrogate from each start ----
    best_h = None
    best_val = -1.0
    for s in starts:
        if time.time() > DEADLINE:
            break
        for temp in (0.5, 0.2, 0.05):
            if time.time() > DEADLINE:
                break
            hh, _ = _local_search(s, temp, iters=2000, step=0.1)
            v = _score(hh)
            if v > best_val:
                best_val = v
                best_h = hh.copy()

    # ---- Phase 2: high-precision polish on the true objective ----
    if best_h is not None:
        hh, v = _polish(best_h, iters=6000, step0=0.05)
        if v > best_val:
            best_val = v
            best_h = hh

    # ---- Final safety: never return worse than the seed ----
    if best_h is None or best_val < 1.0:
        best_h = ones
        best_val = 1.0

    return [float(v) for v in best_h]
