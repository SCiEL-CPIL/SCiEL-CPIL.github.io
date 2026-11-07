"""Heuristic via EXPERT-SELECTION: evaluate a small set of strong bin-packing
heuristics with a fast proxy (greedy + a true ACO simulation) and select the
one that best predicts the fewest bins."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _aco_proxy(demands, capacity, M, n_ants=4, n_iter=6, seed=0):
    """Fast ACO simulation closely mirroring the evaluator's colony.
    Returns the best (minimum) number of bins achieved by any ant.
    """
    n = len(demands)
    d = demands
    C = capacity
    rng = np.random.default_rng(seed)

    # Precompute fitting masks and vectorized selection scores
    fit = (d[None, :] + d[:, None]) <= C          # fit[i, j]
    M_safe = M.copy()
    M_safe[~np.isfinite(M_safe)] = 1e-9
    M_safe[M_safe <= 0] = 1e-9

    # log-sum-exp based selection score (pheromone * heuristic)
    def _select(current, used):
        row = fit[current] & (~used)
        if not np.any(row):
            return -1
        idx = np.nonzero(row)[0]
        tau = np.ones_like(d)
        w = np.log(tau[idx]) + np.log(M_safe[current, idx])
        w -= w.max()
        w = np.exp(w)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return int(idx[0])
        r = rng.random() * s
        c = 0.0
        for k in range(len(idx)):
            c += w[k]
            if r <= c:
                return int(idx[k])
        return int(idx[-1])

    best = n + 1
    for it in range(n_iter):
        iter_best = n + 1
        for a in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 0
            while True:
                rem = np.nonzero(~used)[0]
                if len(rem) == 0:
                    break
                # open a bin: largest remaining item (deterministic, fits)
                s = int(rem[np.argmax(d[rem])])
                used[s] = True
                load = d[s]
                cur = s
                bins += 1
                while True:
                    j = _select(cur, used)
                    if j < 0:
                        break
                    used[j] = True
                    load += d[j]
                    cur = j
            if bins < iter_best:
                iter_best = bins
        if iter_best < best:
            best = iter_best
    return best


def _greedy_proxy(demands, capacity, M, starts=2):
    """Deterministic greedy proxy (fast). Returns mean bins over starts."""
    n = len(demands)
    d = demands
    C = capacity
    start_idx = np.argsort(d)[-starts:]
    total = 0
    for s in start_idx:
        used = np.zeros(n, dtype=bool)
        used[s] = True
        load = d[s]
        cur = s
        bins = 1
        while True:
            rem = np.nonzero(~used)[0]
            fitting = rem[d[rem] <= C - load]
            if len(fitting) == 0:
                break
            j = int(fitting[np.argmax(M[cur, fitting])])
            used[j] = True
            load += d[j]
            cur = j
        rem = np.nonzero(~used)[0]
        if len(rem) == 0:
            break
        s2 = int(rem[np.argmax(d[rem])])
        used[s2] = True
        load = d[s2]
        cur = s2
        bins += 1
    total += bins
    return total / starts


def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 270  # safety margin vs 330s wall clock

    d = np.asarray(demands, dtype=float)
    C = float(capacity)
    n = len(d)

    pair = d[:, None] + d[None, :]
    ok = pair <= C
    empty = C - pair
    ratio = pair / C

    candidates = {}

    # High-power fill ratio (strongly rewards tight fits) - measured improvement
    for p in (3, 5, 8, 10, 12):
        M = np.full((n, n), 1e-9)
        M[ok] = ratio[ok] ** p
        candidates['pow%d' % p] = M

    # Tight fit (1 / (C - sum + eps))
    for eps in (0.1, 1.0, 5.0):
        M = np.full((n, n), 1e-9)
        M[ok] = 1.0 / (empty[ok] + eps)
        candidates['tight_%s' % eps] = M

    # Hybrid: high power ratio + tight
    M = np.full((n, n), 1e-9)
    M[ok] = ratio[ok] ** 8 + 1.0 / (empty[ok] + 1.0)
    candidates['pow8_tight'] = M

    # Gaussian near-completion
    M = np.full((n, n), 1e-9)
    M[ok] = np.exp(-0.1 * (pair[ok] - C) ** 2)
    candidates['gauss'] = M

    # Linear (heavy pairs)
    M = np.full((n, n), 1e-9)
    M[ok] = pair[ok]
    candidates['linear'] = M

    # ---- Evaluate ----
    # First pass: fast greedy to shortlist
    greedy_scores = {}
    for name, M in candidates.items():
        if time.time() > deadline:
            break
        greedy_scores[name] = _greedy_proxy(d, C, M, starts=2)

    # Shortlist top by greedy
    order = sorted(greedy_scores, key=greedy_scores.get)
    shortlist = order[:max(3, len(candidates) // 2)]

    # Second pass: true ACO proxy on the shortlist (closer to real objective)
    aco_scores = {}
    for name in shortlist:
        if time.time() > deadline:
            break
        aco_scores[name] = _aco_proxy(d, C, candidates[name], n_ants=4, n_iter=6)

    # Combine: prefer ACO score; tie-break with greedy
    if aco_scores:
        def _key(name):
            return (aco_scores.get(name, float('inf')),
                    greedy_scores.get(name, float('inf')))
        best_name = min(shortlist, key=_key)
    else:
        best_name = order[0]

    M_best = candidates[best_name].copy()
    M_best[~np.isfinite(M_best)] = 1e-9
    M_best[M_best <= 0] = 1e-9
    return M_best
# EVOLVE-BLOCK-END
