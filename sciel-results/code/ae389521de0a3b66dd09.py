"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin for 150s limit

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    # 1. Initial Solution: Rank Matching
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)

    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")

    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx

    # Vectorized initial cost: sum_{i,j} a[i,j] * b[p[i], p[j]] = sum(a * b[p])
    cost = int(np.sum(a * b[p]))
    best_cost = cost
    best_p = p.copy()

    # Precompute b_rows = b[p] so that b_rows[k] == b[p[k]] is a plain row.
    # After each swap of p[i], p[j], update the two affected rows.
    b_rows = b[p]

    def compute_delta(i, j):
        u = p[i]
        v = p[j]
        # delta = (a[i] - a[j]) @ (b[v, p] - b[u, p]) + 2*a[i,j]*b[u,v]
        # b[v, p] == b_rows[v] (since b_rows[k] = b[p[k]] = b[u] when p[k]=u, etc.)
        diff_a = a[i] - a[j]
        diff_b = b_rows[v] - b_rows[u]
        return int(np.dot(diff_a, diff_b) + 2 * a[i, j] * b[u, v])

    # 2. Local Search: 2-Opt (Pairwise Swap) with random perturbation on stagnation
    while True:
        if time.time() - start_time > time_limit:
            break

        improved = False
        # Sequential first-improvement scan
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            ai = a[i]
            for j in range(i + 1, n):
                delta = compute_delta(i, j)
                if delta < 0:
                    # Perform swap
                    u = p[i]
                    v = p[j]
                    p[i] = v
                    p[j] = u
                    # Update b_rows for the two changed positions
                    b_rows[i] = b[v]
                    b_rows[j] = b[u]
                    cost += delta
                    improved = True
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    # Restart scan from beginning to ensure stability
                    break
            if improved:
                break

        if not improved:
            # Local optimum reached
            elapsed = time.time() - start_time
            if elapsed > time_limit - 5:
                break

            # Perturbation: swap a random pair to escape the local optimum
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                u = p[i]
                v = p[j]
                p[i] = v
                p[j] = u
                b_rows[i] = b[v]
                b_rows[j] = b[u]
                # Recompute cost from scratch for safety after perturbation
                cost = int(np.sum(a * b[p]))
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            else:
                break

    return best_p.tolist()

# EVOLVE-BLOCK-END
