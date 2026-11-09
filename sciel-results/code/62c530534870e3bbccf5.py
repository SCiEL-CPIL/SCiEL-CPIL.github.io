"""Direct heuristic: build a high-quality reference tour, then promote
its edges in the attractiveness matrix so the colony follows a good skeleton."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before the 330s wall-clock kill
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt_vectorized(t, max_passes=500):
        """Vectorized 2-opt: compute all deltas at once for speed."""
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            # Precompute tour as numpy array for fast indexing
            ta = np.array(t)
            # For each pair (i, j) with i < j-1, compute delta
            # delta(i,j) = D[t[i],t[j]] + D[t[i+1],t[(j+1)%n]] - D[t[i],t[i+1]] - D[t[j],t[(j+1)%n]]
            # We can vectorize over j for each i
            found_improvement = False
            for i in range(n - 2):
                if time.time() > deadline:
                    break
                a = t[i]
                b = t[i + 1]
                # For all j from i+2 to n-1
                js = np.arange(i + 2, n)
                cs = np.array([t[j] for j in js])
                ds = np.array([t[(j + 1) % n] for j in js])
                # Compute deltas
                deltas = D[a, cs] + D[b, ds] - D[a, b] - D[cs, ds]
                # Find all improving moves
                improving = deltas < -1e-10
                if np.any(improving):
                    # Apply the best improvement first (largest negative delta)
                    best_idx = np.argmin(deltas)
                    j = js[best_idx]
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
                    found_improvement = True
                    # After one reversal, the tour changed; restart the j loop for this i
                    # But for efficiency, continue checking other j's in the same pass
                    # Actually, after a reversal, the indices change, so we should restart
                    # For simplicity, break and restart the outer loop
                    break
            if not found_improvement:
                # No more improvements in this pass
                break
        return t

    def or_opt_fixed(t, max_passes=100):
        """Fixed Or-opt: relocate single node to better position."""
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n):
                if time.time() > deadline:
                    break
                node = t[i]
                a0 = t[i - 1]
                b0 = t[(i + 1) % n]
                # Cost of removing node from current position
                remove_cost = D[a0, b0]  # new edge replacing a0-node-b0
                # Cost of node's current connections
                cur_cost = D[a0, node] + D[node, b0]
                # Try inserting node between each pair (p, p+1) in the remaining tour
                rest = t[:i] + t[i + 1:]
                best_delta = 0.0
                best_pos = -1
                for p in range(len(rest)):
                    a = rest[p - 1]  # wraps around
                    b = rest[p]
                    # Skip if this is the same position
                    if p == 0 and len(rest) > 0:
                        # Check if inserting at position 0 means between rest[-1] and rest[0]
                        pass
                    insert_cost = D[a, node] + D[node, b] - D[a, b]
                    total_delta = insert_cost + (remove_cost - cur_cost)
                    if total_delta < best_delta - 1e-12:
                        best_delta = total_delta
                        best_pos = p
                if best_pos >= 0 and best_delta < -1e-12:
                    # Insert node at best_pos in rest
                    new_tour = rest[:best_pos] + [node] + rest[best_pos:]
                    t = new_tour
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

    # Build a strong reference tour via randomized multi-start NN + 2-opt (+ Or-opt)
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(0)

    def finish(tour):
        nonlocal best_tour, best_len
        t = two_opt_vectorized(tour)
        t = or_opt_fixed(t)
        t = two_opt_vectorized(t)
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = t
        return t

    # Nearest-neighbor from node 0
    finish(nearest_neighbor(0))
    # Randomized starts
    num_starts = 50
    for _ in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        s = int(rng.integers(n))
        finish(nearest_neighbor(s))

    # Cheap random-restart 2-opt refinement of the best tour
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        t = list(best_tour)
        # Apply 2-3 random reversals
        for _ in range(3):
            i, j = int(rng.integers(n)), int(rng.integers(n))
            if i > j:
                i, j = j, i
            if j - i >= 1:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        t = two_opt_vectorized(t)
        finish(t)

    if best_tour is None:
        best_tour = nearest_neighbor(0)
        best_tour = two_opt_vectorized(best_tour)

    # Build the attractiveness matrix: inverse distance, boosted on reference-tour edges.
    diag = np.eye(n, dtype=bool)
    safe = D.copy()
    safe[diag] = 1.0
    base = 1.0 / safe

    # Promote the reference tour's directed edges with a moderate factor.
    # Use a factor that's strong but not overwhelming, allowing some exploration
    for k in range(n):
        i = best_tour[k]
        j = best_tour[(k + 1) % n]
        base[i, j] = 1.0e3

    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
# EVOLVE-BLOCK-END
