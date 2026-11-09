# EVOLVE-BLOCK-START
import numpy as np
import time

def _setup(n=64, M=20000):
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    # Precompute the ordering of L_i at x=0 (gaps = (a_j-a_i)*y >= 0, so order = order of a)
    perm = np.argsort(a)  # permutation p: L_sorted[k] = L_{perm[k]}
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    # For each consecutive sorted pair (k, k+1): lower index, upper index
    lo = perm[:-1]
    hi = perm[1:]
    return y, a, w, perm, inv, lo, hi

def _union_area_fast(x, y, a, w, lo, hi, n=64, M=20000):
    """O(M) area using precomputed permutation (valid when x differences are small)."""
    # L at the lower and upper of each pair, for all y
    Llo = x[lo][:, np.newaxis] + a[lo][:, np.newaxis] * y[np.newaxis, :]
    Lhi = x[hi][:, np.newaxis] + a[hi][:, np.newaxis] * y[np.newaxis, :]
    d = Lhi - Llo
    union = w[np.newaxis, :] + np.sum(np.minimum(d, w[np.newaxis, :]), axis=0)
    return float(np.mean(union))

def _union_area_grad(x, y, a, w, lo, hi, n=64, M=20000):
    """Return (area, subgradient) in O(M)."""
    Llo = x[lo][:, np.newaxis] + a[lo][:, np.newaxis] * y[np.newaxis, :]
    Lhi = x[hi][:, np.newaxis] + a[hi][:, np.newaxis] * y[np.newaxis, :]
    d = Lhi - Llo
    sat = (d < w[np.newaxis, :])  # (n-1, M)
    union = w[np.newaxis, :] + np.sum(np.minimum(d, w[np.newaxis, :]), axis=0)
    area = float(np.mean(union))
    # subgradient: for each pair, +1 to upper, -1 to lower, where saturated
    g = np.zeros(n, dtype=np.float64)
    g[hi] += np.sum(sat, axis=1) / M
    g[lo] -= np.sum(sat, axis=1) / M
    return area, g

def _union_area_ref(x, n=64, M=20000):
    """Reference O(M n log n) area, used only for final validation."""
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    union = w[np.newaxis, :] + np.sum(np.minimum(d, w[np.newaxis, :]), axis=0)
    return float(np.mean(union))

def _local_search(x, y, a, w, lo, hi, n=64, M=20000, deadline=None, max_iters=4000):
    """Pairwise swap local search."""
    best = x.copy()
    best_a = _union_area_fast(best, y, a, w, lo, hi, n, M)
    it = 0
    while it < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # vectorized: try swapping i and j for all i<j
        # For a swap, the area change can be evaluated; we do a greedy pass.
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return best, best_a
            for j in range(i + 1, n):
                trial = best.copy()
                trial[i], trial[j] = best[j], best[i]
                ta = _union_area_fast(trial, y, a, w, lo, hi, n, M)
                if ta < best_a - 1e-15:
                    best = trial
                    best_a = ta
                    improved = True
        it += 1
        if not improved:
            break
    return best, best_a

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin before 630s kill

    y, a, w, perm, inv, lo, hi = _setup(n, M)

    # Seed: constant zero
    best_x = np.zeros(n, dtype=np.float64)
    best_area = _union_area_fast(best_x, y, a, w, lo, hi, n, M)

    def subgrad_descent(x0, deadline):
        x = x0.copy()
        area, g = _union_area_grad(x, y, a, w, lo, hi, n, M)
        # Polyak: step = <x - x*, g> / ||g||^2 ; use x* ~ 0
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-12:
            return x, area
        # momentum
        v = np.zeros(n, dtype=np.float64)
        mu = 0.9
        # Polyak step scale
        step_polyak = max(np.dot(x, g), 1e-9) / (gnorm * gnorm + 1e-12)
        step = min(step_polyak, 0.5)
        it = 0
        while time.time() < deadline:
            it += 1
            v = mu * v - step * g
            x_new = x + v
            a_new, g_new = _union_area_grad(x_new, y, a, w, lo, hi, n, M)
            if a_new < area:
                x = x_new
                area = a_new
                g = g_new
                gnorm = np.linalg.norm(g)
                if gnorm < 1e-12:
                    break
                step_polyak = max(np.dot(x, g), 1e-9) / (gnorm * gnorm + 1e-12)
                step = min(step_polyak, 0.5)
            else:
                # reduce step
                step *= 0.7
                v *= 0.5
            if it % 50 == 0 and step < 1e-9:
                break
        return x, area

    # Phase 1: subgradient descent from a few random restarts
    rng = np.random.default_rng(12345)
    n_restarts = 8
    per_restart = max(0.5, (deadline - start_time) / n_restarts - 5.0)
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        if r == 0:
            x0 = np.zeros(n, dtype=np.float64)
        else:
            x0 = rng.normal(0.0, 0.5, size=n)
        rd = min(deadline - 15.0, time.time() + per_restart)
        x_r, a_r = subgrad_descent(x0, rd)
        if a_r < best_area:
            best_x = x_r.copy()
            best_area = a_r

    # Phase 2: local search polish
    if time.time() < deadline - 20:
        best_x, best_area = _local_search(best_x, y, a, w, lo, hi, n, M,
                                          deadline=deadline - 10, max_iters=2000)

    # Phase 3: a couple more short subgradient passes from the polished point
    if time.time() < deadline - 10:
        for r in range(2):
            if time.time() > deadline - 10:
                break
            rd = min(deadline - 10, time.time() + 8.0)
            x_r, a_r = subgrad_descent(best_x + rng.normal(0, 0.1, n), rd)
            if a_r < best_area:
                best_x = x_r.copy()
                best_area = a_r
                if time.time() < deadline - 10:
                    best_x, best_area = _local_search(best_x, y, a, w, lo, hi, n, M,
                                                      deadline=deadline - 10, max_iters=500)

    # Final validation with reference
    if not np.all(np.isfinite(best_x)):
        return np.zeros(n, dtype=np.float64)
    ref_area = _union_area_ref(best_x, n, M)
    if not np.isfinite(ref_area) or ref_area <= 0:
        return np.zeros(n, dtype=np.float64)

    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
