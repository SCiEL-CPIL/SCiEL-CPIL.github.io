"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute difference matrices for O(1)-ish delta per swap
    # A_row_diff[i, j, :] = a[i, :] - a[j, :]  -> shape (n, n, n) is too big (1M*8=8MB, ok)
    # But we can avoid storing all: compute on the fly per i, j pair in vectorized way.
    # Better: precompute a_row_diff as (n, n, n) float64 = 100^3 * 8 = 8 MB, fine.
    # Actually let's use a different approach: precompute for each location pair (u,v)
    # the vector b_diff[u, v, :] = b[v, :] - b[u, :]  shape (n, n, n) = 8 MB
    # And a_row_diff[i, j, :] = a[i, :] - a[j, :] shape (n, n, n) = 8 MB
    # Total 16 MB, acceptable.

    # b_diff[u, v, k] = b[v, k] - b[u, k]
    b_diff = b[None, :, :] - b[:, None, :]  # (n, n, n): b_diff[u,v,k] = b[v,k]-b[u,k]
    # a_row_diff[i, j, k] = a[i, k] - a[j, k]
    a_row_diff = a[None, :, :] - a[:, None, :]  # (n, n, n): a_row_diff[i,j,k] = a[i,k]-a[j,k]
    # a_col_diff[i, j, k] = a[k, i] - a[k, j]
    a_col_diff = a[:, None, :] - a[None, :, :]  # (n, n, n): a_col_diff[i,j,k] = a[k,i]-a[k,j]

    def compute_cost(p):
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    # Initial solution: greedy rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    best_p = p.copy()
    best_cost = compute_cost(p)

    def compute_delta_swap(p, i, j):
        u = p[i]
        v = p[j]
        # delta = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
        #       + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        #       + (a[i,j]-a[j,i])*(b[v,u]-b[u,v])
        # Using precomputed:
        # term1 = sum_k a_row_diff[i,j,k] * b_diff[u,v, p[k]]  for k != i,j
        # term2 = sum_k a_col_diff[i,j,k] * b_diff[p[k],u,v]  ... wait
        # b[p[k], v] - b[p[k], u] = b_diff[p[k], u, v]? No.
        # b_diff[x, y, z] = b[y, z] - b[x, z]
        # So b[v, p[k]] - b[u, p[k]] = b_diff[u, v, p[k]]
        # And b[p[k], v] - b[p[k], u] = b_diff[u, v, ...] no.
        # b[p[k], v] - b[p[k], u]: this is b[location=p[k], col=v] - b[location=p[k], col=u]
        # = b_diff[u, v, ...] doesn't directly give this.
        # Let's define b_col_diff[u, v, k] = b[k, v] - b[k, u]
        # That's just b_diff.T or b[:, :, v] - b[:, :, u]
        # Actually b_col_diff[u, v, k] = b[k, v] - b[k, u] = b_diff.T[v, u, k] if b_diff[u,v,k]=b[v,k]-b[u,k]
        # b_diff.T[v, u, k] = b_diff[k, v, u] = b[u, k] - b[v, k] ... no that's wrong sign.
        # Let me just precompute both.

        # Let's just use the mask approach but vectorize the whole pass at once.
        # For a given p, compute ALL deltas at once:
        # For each (i,j) pair, delta_ij depends on p.
        # This is complex. Let's stick with per-pair but use precomputed b_diff.

        # term1: sum_{k!=i,j} a_row_diff[i,j,k] * b_diff[u, v, p[k]]
        # term2: sum_{k!=i,j} a_col_diff[i,j,k] * (b[p[k], v] - b[p[k], u])
        # b[p[k], v] - b[p[k], u]: precompute b_col_diff[u, v, k] = b[k, v] - b[k, u]
        # Then term2 = sum_{k!=i,j} a_col_diff[i,j,k] * b_col_diff[u, v, p[k]]

        # Precompute b_col_diff: b_col_diff[u, v, k] = b[k, v] - b[k, u]
        # b_col_diff = b[:, None, None, :] ... no, shape (n, n, n)
        # b_col_diff[u, v, k] = b[k, v] - b[k, u]
        # = b.T[v, k] - b.T[u, k] ... no
        # b[k, v] is b[k, v], b[k, u] is b[k, u]
        # So b_col_diff[u, v, :] = b[:, v] - b[:, u]  -> shape (n,) for each (u,v)
        # b_col_diff = b[:, None, :] - b[None, :, :]  # (n, n, n): b_col_diff[u,v,k] = b[k,v]-b[k,u]? 
        # b[:, None, :] has shape (n, 1, n), b[None, :, :] has shape (1, n, n)
        # b[:, None, :] - b[None, :, :] -> (n, n, n) where element [k, v, u] = b[k, v] - b[k, u]
        # So b_col_diff[k, v, u] = b[k, v] - b[k, u]
        # We want b_col_diff[u, v, k] = b[k, v] - b[k, u] = above[k, v, u]
        # So b_col_diff = (b[:, None, :] - b[None, :, :]).transpose(1, 2, 0)  # shape (n,n,n)
        # b_col_diff[u, v, k] = b[k, v] - b[k, u] ✓

        # This is getting complex. Let's just use the mask approach but optimize it.
        # The mask creation is the bottleneck. Let's precompute a list of indices excluding i,j.
        # Actually for n=100, the mask approach with fancy indexing is probably ~5-10μs per call.
        # 4950 pairs * 10μs = 50ms per pass. With 50 restarts and multiple passes, that's manageable.
        # Let's just optimize the inner loop.

        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False

        # term1
        diff_a = a_row_diff[i, j, mask]  # a[i,k]-a[j,k] for k in mask
        diff_b1 = b[v, p[mask]] - b[u, p[mask]]  # b[v,p[k]]-b[u,p[k]]
        term1 = np.dot(diff_a, diff_b1)

        # term2
        diff_a2 = a[mask, i] - a[mask, j]  # a[k,i]-a[k,j]
        diff_b2 = b[p[mask], v] - b[p[mask], u]  # b[p[k],v]-b[p[k],u]
        term2 = np.dot(diff_a2, diff_b2)

        term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])

        return term1 + term2 + term3

    # Precompute all-pairs index lists to avoid mask creation
    # For each (i,j), store the indices excluding i and j
    # This is 4950 * 98 indices = ~485k ints, fine
    pair_indices = {}
    for i in range(n):
        for j in range(i + 1, n):
            idx = [k for k in range(n) if k != i and k != j]
            pair_indices[(i, j)] = np.array(idx, dtype=np.int64)

    def compute_delta_swap_fast(p, i, j):
        u = p[i]
        v = p[j]
        idx = pair_indices[(i, j)]
        pm = p[idx]

        # term1: sum_k (a[i,k]-a[j,k]) * (b[v,p[k]]-b[u,p[k]])
        term1 = np.dot(a[i, idx] - a[j, idx], b[v, pm] - b[u, pm])

        # term2: sum_k (a[k,i]-a[k,j]) * (b[p[k],v]-b[p[k],u])
        term2 = np.dot(a[idx, i] - a[idx, j], b[pm, v] - b[pm, u])

        term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])

        return term1 + term2 + term3

    # Hill climbing with random restarts
    np.random.seed(42)
    for restart in range(80):
        if time.time() > deadline:
            break

        if restart == 0:
            p = best_p.copy()
        else:
            p = best_p.copy()
            num_swaps = min(3 + restart // 15, 15)
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # First pass: find all improving swaps, apply best
            best_delta = 0
            best_pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    d = compute_delta_swap_fast(p, i, j)
                    if d < best_delta:
                        best_delta = d
                        best_pair = (i, j)
            if best_pair is not None:
                i, j = best_pair
                p[i], p[j] = p[j], p[i]
                improved = True

            if improved:
                cur_cost = compute_cost(p)
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_p = p.copy()

    # Final polish: exhaustive 2-opt until no improvement
    if time.time() < deadline:
        p = best_p.copy()
        while time.time() < deadline:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    d = compute_delta_swap_fast(p, i, j)
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        improved = True
            if not improved:
                break
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
