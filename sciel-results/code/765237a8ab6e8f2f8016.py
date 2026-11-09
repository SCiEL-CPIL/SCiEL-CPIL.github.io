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
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                    delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                    if delta < -1e-12:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
        return t

    def or_opt(t, max_passes=200):
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n):
                if not improved and time.time() > deadline:
                    break
                # relocate single node i to a better position
                node = t[i]
                rest = t[:i] + t[i + 1:]
                cur = D[t[i - 1], node] + D[node, t[(i + 1) % n]]
                best_delta = -1e-12
                best_pos = -1
                for p in range(len(rest)):
                    a, b = rest[p - 1], rest[p]
                    new = D[a, node] + D[node, b] - (D[a, b])
                    if new - (D[a, b] - D[a, b]) < 0:  # placeholder, replaced below
                        pass
                # proper evaluation
                best_delta = 0.0
                best_pos = -1
                for p in range(len(rest)):
                    a, b = rest[p - 1], rest[p]
                    cand = D[a, node] + D[node, b] - D[a, b]
                    if cand < cur + best_delta - 1e-12 and (best_pos < 0 or cand < best_delta + cur):
                        # we want minimal total change; track absolute new length
                        pass
                # Simpler: compute new tour length if moved to each position is O(n^2);
                # do a cheaper delta: removing node costs -(D[a0,node]+D[node,b0]) where
                # a0,b0 are its old neighbors; inserting between a,b costs +(D[a,node]+D[node,b]-D[a,b]).
                a0, b0 = t[i - 1], t[(i + 1) % n]
                remove_cost = -(D[a0, node] + D[node, b0])
                best_insert = None
                best_val = float('inf')
                for p in range(len(rest)):
                    a, b = rest[p - 1], rest[p]
                    if a == a0 and b == b0:
                        continue
                    val = D[a, node] + D[node, b] - D[a, b]
                    if val < best_val:
                        best_val = val
                        best_insert = p
                if best_insert is not None and (remove_cost + best_val) < -1e-12:
                    rest2 = rest[:best_insert] + [node] + rest[best_insert:]
                    t = rest2
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

    # Build a pool of high-quality tours to establish a consensus heuristic
    rng = np.random.default_rng(0)
    pool = []
    
    def add_to_pool(tour):
        t = two_opt(tour)
        # Ensure uniqueness to avoid biasing frequency with duplicates
        # Simple check: tuple of sorted nodes? No, tour order matters for directed edges.
        # Just add it. Duplicates are fine, they reinforce the edge.
        pool.append(t)
        return t

    # 1. Nearest-neighbor from all nodes (or a subset if time is tight, but n=50 is fast)
    # Let's do 100 random starts to get diversity
    num_starts = 100
    for _ in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        s = int(rng.integers(n))
        add_to_pool(nearest_neighbor(s))
        
    # 2. Random restarts from the best tour found so far to explore local neighborhoods
    if pool:
        # Find the best in the pool
        best_t = min(pool, key=tour_len)
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            t = list(best_t)
            # Apply random 2-opt moves
            for _ in range(5):
                i, j = int(rng.integers(n)), int(rng.integers(n))
                if i > j:
                    i, j = j, i
                if j - i >= 1:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
            add_to_pool(t)

    if not pool:
        # Fallback
        pool.append(two_opt(nearest_neighbor(0)))

    # Build the attractiveness matrix based on edge frequency in the pool
    # H_ij = (1 / d_ij) * (1 + freq_ij)
    diag = np.eye(n, dtype=bool)
    safe = D.copy()
    safe[diag] = 1.0
    base = 1.0 / safe
    
    freq = np.zeros((n, n), dtype=float)
    for t in pool:
        for k in range(n):
            i = t[k]
            j = t[(k + 1) % n]
            freq[i, j] += 1.0
            
    # Scale the frequency contribution
    # If freq is high, the edge is strongly preferred.
    # base is ~1/d. freq is ~count.
    # We want the heuristic to be dominated by the consensus for good edges.
    # Let's use base * (1 + freq)
    base = base * (1.0 + freq)
    
    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
# EVOLVE-BLOCK-END
