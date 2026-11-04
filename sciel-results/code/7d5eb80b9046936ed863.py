"""QAP tai256c solver: Sinkhorn-initialized iterated local search with vectorized swap deltas."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        """Compute cost delta for swapping facilities at positions u and v."""
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search(p, cost, max_passes=100):
        """Best-improvement local search using swap deltas."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                return p, cost

            best_delta = 0
            best_i = -1
            best_j = -1

            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    d = fast_swap_delta(p, i, j)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j

            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True

        return p, cost

    def sinkhorn_init(eps=1.0, max_iter=200):
        """
        Sinkhorn-Knopp scaling to find a soft assignment, then extract a permutation.
        We scale a matrix derived from a and b to approximate the optimal assignment.
        """
        if time.time() > deadline:
            return np.random.permutation(n).astype(np.int64)

        # Build a cost matrix for the linear assignment relaxation
        # C[i,j] = sum_k a[i,k] * b[j,k]  (approximate linearization)
        # Actually, use the standard linearization:
        # C[i,j] = sum_k a[i,k] * b[j,k] + sum_k a[k,i] * b[k,j]
        C = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            C[i, :] = a[i, :] @ b[:, :]  # a[i,:] dot b[j,:] for each j
            C[i, :] += a[:, i] @ b[:, :]  # a[k,i] dot b[k,j] for each j

        # Sinkhorn-Knopp: find doubly stochastic matrix P minimizing <P, C>
        # P = D1 * exp(-C/eps) * D2
        # Iterate: scale rows to sum to 1, then columns to sum to 1
        M = np.exp(-C / eps)
        # Ensure no zeros
        M = np.maximum(M, 1e-10)

        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Scale rows
            row_sums = M.sum(axis=1, keepdims=True)
            M = M / row_sums
            # Scale columns
            col_sums = M.sum(axis=0, keepdims=True)
            M = M / col_sums

        # Extract permutation from the soft assignment using Hungarian
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2
        C = M1 + M2
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    # --- Initial solutions ---
    # 1. Rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central

    # 2. Sinkhorn initialization
    p_sink = sinkhorn_init(eps=1.0, max_iter=100)

    # 3. Random
    p_rand = np.random.permutation(n).astype(np.int64)

    # Evaluate all initial solutions
    candidates = [p_rank, p_sink, p_rand]
    costs = [compute_cost(p) for p in candidates]

    # Pick the best initial solution
    best_idx = int(np.argmin(costs))
    current_p = candidates[best_idx].copy()
    current_cost = costs[best_idx]

    # Local search from the best initial
    current_p, current_cost = local_search(current_p, current_cost)
    best_p = current_p.copy()
    best_cost = current_cost

    # Also try local search from other competitive initial solutions
    for idx in range(3):
        if idx == best_idx:
            continue
        if costs[idx] < best_cost * 1.05:  # within 5% of best
            p_try = candidates[idx].copy()
            p_try, c_try = local_search(p_try, costs[idx])
            if c_try < best_cost:
                best_p = p_try.copy()
                best_cost = c_try

    # --- Iterated Local Search ---
    iteration = 0
    while time.time() < deadline - 30:
        iteration += 1

        # Perturbation: mix of strategies
        p = best_p.copy()
        strategy = iteration % 5
        if strategy == 0:
            # Random swaps
            n_swaps = np.random.randint(3, 8)
            for _ in range(n_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
        elif strategy == 1:
            # Block reversal (2-opt)
            i = np.random.randint(0, n - 2)
            j = np.random.randint(i + 2, n)
            p[i:j+1] = p[i:j+1][::-1]
        elif strategy == 2:
            # Or-opt: move a small block
            block_len = np.random.randint(1, 4)
            i = np.random.randint(0, n - block_len)
            j = np.random.randint(0, n)
            block = p[i:i+block_len].copy()
            # Remove block
            p = np.delete(p, slice(i, i+block_len))
            # Insert at j (adjust index if needed)
            if j > i:
                j = j - block_len
            j = min(j, len(p))
            p = np.insert(p, j, block)
        elif strategy == 3:
            # Larger perturbation
            n_swaps = np.random.randint(8, 15)
            for _ in range(n_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
        else:
            # Hungarian re-optimization as perturbation
            h_p = hungarian_reopt(p)
            p = h_p.copy()

        # Local search from perturbed solution
        p_cost = compute_cost(p)
        p, p_cost = local_search(p, p_cost)

        if p_cost < best_cost:
            best_p = p.copy()
            best_cost = p_cost

    # Final polish
    if time.time() < deadline - 2:
        final_p, final_cost = local_search(best_p.copy(), best_cost, max_passes=200)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost

    return best_p.tolist()
