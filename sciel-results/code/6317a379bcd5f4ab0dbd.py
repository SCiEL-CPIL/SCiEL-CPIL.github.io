"""Stronger heuristic: build a high-quality reference tour, then boost its edges."""
import numpy as np
import time


def _tour_length(tour, dm):
    t = np.asarray(tour)
    return float(dm[t, np.roll(t, -1)].sum())


def _two_opt(tour, dm, time_budget=None):
    """Improve a closed tour with 2-opt; returns (best_tour, best_len)."""
    n = len(tour)
    tour = np.array(tour, dtype=int)
    best = _tour_length(tour, dm)
    if n < 4:
        return tour, best
    if time_budget is not None:
        t0 = time.time()
    while True:
        improved = False
        for i in range(n - 1):
            if time_budget is not None and (time.time() - t0) > time_budget:
                return tour, best
            a = tour[i]
            b = tour[i + 1]
            d_ab = dm[a, b]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # would be a no-op (reversing the whole tour)
                c = tour[j]
                d = tour[(j + 1) % n]
                d_cd = dm[c, d]
                d_ac = dm[a, c]
                d_bd = dm[b, d]
                if d_ac + d_bd < d_ab + d_cd - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    best = best - (d_ab + d_cd) + d_ac + d_bd
                    improved = True
        if not improved:
            break
    return tour, best


def heuristics(distance_matrix):
    start_time = time.time()
    HARD_LIMIT = 330.0
    SAFETY = 50.0
    budget = HARD_LIMIT - SAFETY - (time.time() - start_time)
    if budget <= 1.0:
        budget = 1.0

    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    if n < 2:
        return np.ones_like(dm)

    # Baseline heuristic: 1/d (valid, positive, finite).
    base = 1.0 / (dm + 1e-9)

    # Build diverse initial tours.
    def nn_tour(start):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        cur = start
        for _ in range(n - 1):
            d = dm[cur].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return tour

    inits = []
    # Nearest-neighbor from every start node.
    for s in range(n):
        inits.append(nn_tour(s))
    # A few random permutations (deterministic seed for reproducibility).
    rng = np.random.default_rng(12345)
    for _ in range(6):
        inits.append(list(rng.permutation(n)))

    # Improve each with 2-opt under a shared time budget, keep the best.
    per_init = max(0.05, budget / max(len(inits), 1))
    best_tour = None
    best_len = np.inf
    for init in inits:
        if time.time() - start_time > budget:
            break
        t0 = time.time()
        rem = max(0.05, budget - (time.time() - start_time))
        t, L = _two_opt(init, dm, time_budget=min(per_init, rem))
        if L < best_len:
            best_len = L
            best_tour = t

    if best_tour is None:
        # Fallback: use a single NN tour (already decent) without 2-opt if no time.
        best_tour = nn_tour(0)

    # Encode the reference tour into the heuristic: strongly boost its edges.
    h = base.copy()
    t = np.asarray(best_tour, dtype=int)
    for k in range(n):
        i = t[k]
        j = t[(k + 1) % n]
        h[i, j] = base[i, j] * 100.0

    # Safety: ensure positive, finite, correct shape.
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    if h.shape != dm.shape:
        h = base
    return h
