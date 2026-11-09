"""Direct heuristic: build a high-quality reference tour, then strongly promote
its edges in the attractiveness matrix so the colony follows a good skeleton."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before the 330s wall-clock kill
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=200):
        """Vectorized 2-opt: compute all deltas at once per pass."""
        t = np.asarray(t, dtype=int)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            # Build index arrays for all (i,j) pairs
            i_idx = np.arange(n - 1)
            # For each i, j ranges from i+2 to n-1 (skip i=0,j=n-1)
            # We'll iterate per-i but vectorize over j
            for i in range(n - 1):
                if i == 0:
                    j_range = range(i + 2, n - 1)  # exclude j=n-1 when i=0
                else:
                    j_range = range(i + 2, n)
                if len(j_range) == 0:
                    continue
                js = np.array(list(j_range), dtype=int)
                a = t[i]
                b = t[i + 1]
                cs = t[js]
                ds = t[(js + 1) % n]
                deltas = (D[a, cs] + D[b, ds]) - (D[a, b] + D[cs, ds])
                # Find first improving move
                mask = deltas < -1e-12
                if np.any(mask):
                    j = js[np.argmax(mask)]
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
        return t

    def or_opt(t, max_passes=50):
        """Or-opt: relocate single node to best position."""
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for k in range(n):
                node = t[k]
                prev = t[(k - 1) % n]
                nxt = t[(k + 1) % n]
                cost_remove = D[prev, node] + D[node, nxt] - D[prev, nxt]
                # Try inserting node between every pair (a,b)
                best_delta = cost_remove
                best_pos = -1
                for pos in range(n - 1):
                    if pos == k or pos == (k - 1) % n:
                        continue
                    a = t[pos]
                    b = t[(pos + 1) % n]
                    if a == prev or b == prev or a == nxt or b == nxt:
                        continue
                    cost_insert = D[a, node] + D[node, b] - D[a, b]
                    delta = cost_remove + cost_insert
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_pos = pos
                if best_pos >= 0:
                    t.pop(k)
                    t.insert(best_pos + 1, node)
                    improved = True
        return t

    def nearest_neighbor(start):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        cur = start
        for _ in range(n - 1):
            d = D[cur, :].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return tour

    # Build a strong reference tour via randomized multi-start NN + 2-opt + or-opt
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(0)

    def finish(tour):
        nonlocal best_tour, best_len
        t = two_opt(tour)
        t = or_opt(t)
        t = two_opt(t)
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = t
        return t

    # Nearest-neighbor from node 0
    finish(nearest_neighbor(0))
    # Randomized starts - more starts now that 2-opt is faster
    num_starts = 120
    for _ in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        s = int(rng.integers(n))
        finish(nearest_neighbor(s))

    # Random-restart 2-opt + or-opt refinement of the best tour
    for _ in range(40):
        if time.time() > deadline - 5.0:
            break
        t = list(best_tour)
        for _ in range(3):
            i, j = int(rng.integers(n)), int(rng.integers(n))
            if i > j:
                i, j = j, i
            if j - i >= 1:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        t = two_opt(t)
        t = or_opt(t)
        t = two_opt(t)
        finish(t)

    if best_tour is None:
        best_tour = nearest_neighbor(0)
        best_tour = two_opt(best_tour)

    # Build the attractiveness matrix: inverse distance, boosted on reference-tour edges.
    diag = np.eye(n, dtype=bool)
    safe = D.copy()
    safe[diag] = 1.0
    base = 1.0 / safe

    # Promote the reference tour's directed edges with a large factor.
    for k in range(n):
        i = best_tour[k]
        j = best_tour[(k + 1) % n]
        base[i, j] = 1.0e5

    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
# EVOLVE-BLOCK-END
