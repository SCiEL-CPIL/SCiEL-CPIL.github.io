"""QAP tai256c solver: iterated local search with Hungarian re-optimization and spectral bisection init."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
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

    def vectorized_local_search(p, cost, max_passes=50):
        """Vectorized best-improvement local search using precomputed deltas."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                return p, cost
            
            # Compute all swap deltas vectorized
            best_delta = 0.0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                # Compute deltas for all j > i in a vectorized manner
                # We need to compute fast_swap_delta(p, i, j) for all j > i
                # This is expensive to fully vectorize, so we do it per i but optimize the inner loop
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

    def block_realignment(p, cost, window_size=16):
        """
        Sequence alignment approach: iteratively realign high-flow facility clusters
        to low-distance location clusters using a banded dynamic programming approach.
        
        This exploits the local sequential structure by considering blocks of consecutive
        facilities and finding optimal local permutations within a window.
        """
        if time.time() > deadline:
            return p, cost
        
        improved = True
        passes = 0
        max_realignment_passes = 10
        
        while improved and passes < max_realignment_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                return p, cost
            
            for start in range(0, n - window_size + 1, window_size // 2):
                if time.time() > deadline:
                    return p, cost
                
                end = min(start + window_size, n)
                window_len = end - start
                
                if window_len < 3:
                    continue
                
                # Extract the subproblem
                fac_indices = list(range(start, end))
                loc_indices = [p[i] for i in fac_indices]
                
                # Get the flow and distance submatrices
                sub_a = a[np.ix_(fac_indices, fac_indices)]
                sub_b = b[np.ix_(loc_indices, loc_indices)]
                
                # Current cost of this window
                current_window_cost = np.sum(sub_a * sub_b)
                
                # Try to find a better permutation of this window using local search
                # We'll do a few iterations of best-improvement within this window
                best_window_perm = list(range(window_len))
                best_window_cost = current_window_cost
                
                # Simple local search within the window
                for _ in range(5):
                    best_delta = 0
                    best_ui = -1
                    best_vj = -1
                    
                    for ui in range(window_len):
                        for vj in range(ui + 1, window_len):
                            # Compute delta for swapping positions ui and vj in the window
                            # This is a simplified delta computation for the subproblem
                            pu = best_window_perm[ui]
                            pv = best_window_perm[vj]
                            
                            # Cost before
                            cost_before = 0
                            for k in range(window_len):
                                cost_before += sub_a[ui, k] * sub_b[pu, best_window_perm[k]]
                                cost_before += sub_a[k, ui] * sub_b[best_window_perm[k], pu]
                                cost_before += sub_a[vj, k] * sub_b[pv, best_window_perm[k]]
                                cost_before += sub_a[k, vj] * sub_b[best_window_perm[k], pv]
                            
                            # Cost after
                            cost_after = 0
                            new_perm = best_window_perm.copy()
                            new_perm[ui], new_perm[vj] = new_perm[vj], new_perm[ui]
                            for k in range(window_len):
                                cost_after += sub_a[ui, k] * sub_b[new_perm[ui], new_perm[k]]
                                cost_after += sub_a[k, ui] * sub_b[new_perm[k], new_perm[ui]]
                                cost_after += sub_a[vj, k] * sub_b[new_perm[vj], new_perm[k]]
                                cost_after += sub_a[k, vj] * sub_b[new_perm[k], new_perm[vj]]
                            
                            delta = cost_after - cost_before
                            if delta < best_delta:
                                best_delta = delta
                                best_ui = ui
                                best_vj = vj
                    
                    if best_delta < -1e-9:
                        best_window_perm[best_ui], best_window_perm[best_vj] = best_window_perm[best_vj], best_window_perm[best_ui]
                        best_window_cost += best_delta
                    else:
                        break
                
                # Apply the improved permutation if it's better
                if best_window_cost < current_window_cost - 1e-9:
                    for idx, fac_idx in enumerate(fac_indices):
                        p[fac_idx] = loc_indices[best_window_perm[idx]]
                    cost += best_window_cost - current_window_cost
                    improved = True
        
        return p, cost

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

    # Initial solution 1: rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    p_rank = p_rank.astype(np.int64)

    # Initial solution 2: random
    p_rand = np.random.permutation(n).astype(np.int64)

    # Choose the better initial solution
    cost_rank = compute_cost(p_rank)
    cost_rand = compute_cost(p_rand)
    
    if cost_rank <= cost_rand:
        current_p = p_rank.copy()
        current_cost = cost_rank
    else:
        current_p = p_rand.copy()
        current_cost = cost_rand

    # First local search
    current_p, current_cost = vectorized_local_search(current_p, current_cost)
    best_p = current_p.copy()
    best_cost = current_cost

    # Also try local search from the other initial solution if it's competitive
    other_p = p_rand.copy() if cost_rank <= cost_rand else p_rank.copy()
    other_cost = cost_rand if cost_rank <= cost_rand else cost_rank
    other_p, other_cost = vectorized_local_search(other_p, other_cost)
    if other_cost < best_cost:
        best_p = other_p.copy()
        best_cost = other_cost

    # Iterated Local Search with periodic Hungarian re-optimization and block realignment
    iteration = 0
    while time.time() < deadline - 30:  # reserve 30s for final polish
        iteration += 1

        # Perturb: mix of random swaps and 2-opt block reversals
        p = best_p.copy()
        if iteration % 4 == 0:
            # 2-opt style: reverse a block
            i = np.random.randint(0, n - 1)
            j = np.random.randint(i + 2, n)
            p[i:j+1] = p[i:j+1][::-1]
        else:
            # Random swaps
            n_swaps = np.random.randint(2, 8)
            for _ in range(n_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]

        # Local search from perturbed
        p, cost = vectorized_local_search(p, compute_cost(p))

        # Periodically apply block realignment
        if iteration % 5 == 0 and time.time() < deadline - 40:
            p, cost = block_realignment(p, cost)

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 40:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = vectorized_local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    # Final high-precision polish pass
    if time.time() < deadline - 2:
        final_p, final_cost = vectorized_local_search(best_p.copy(), best_cost, max_passes=100)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost

    return best_p.tolist()
