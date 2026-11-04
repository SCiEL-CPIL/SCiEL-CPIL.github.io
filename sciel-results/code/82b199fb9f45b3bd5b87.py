"""QAP tai256c solver: iterated local search with vectorized swap deltas and Hungarian re-optimization."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
import random


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

    def vectorized_local_search(p, cost, max_passes=100):
        """First-improvement local search: scan swaps in random order, apply first negative delta."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                return p, cost
            
            # Generate random order of pairs to avoid bias
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    return p, cost
                
                for j in order:
                    if j <= i:
                        continue
                    d = fast_swap_delta(p, i, j)
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += d
                        improved = True
                        break
                if improved:
                    # Restart scan after a swap
                    break
        return p, cost

    def block_realignment(p, cost, window_size=12):
        """
        Block realignment: for each window of consecutive facilities, try to find
        a better assignment of locations using Hungarian algorithm on the subproblem.
        """
        if time.time() > deadline:
            return p, cost
        
        improved = True
        passes = 0
        max_realignment_passes = 5
        
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
                
                fac_indices = np.arange(start, end)
                loc_indices = p[fac_indices]
                
                # Get the flow and distance submatrices
                sub_a = a[np.ix_(fac_indices, fac_indices)]
                sub_b = b[np.ix_(loc_indices, loc_indices)]
                
                # Current cost of this window
                current_window_cost = np.sum(sub_a * sub_b)
                
                # Solve the sub-QAP using Hungarian on a linearized cost matrix
                C = sub_a @ sub_b.T + sub_a.T @ sub_b
                row_ind, col_ind = linear_sum_assignment(C)
                new_window_perm = col_ind
                
                # Check if this gives a better cost
                new_window_cost = np.sum(sub_a * sub_b[np.ix_(new_window_perm, new_window_perm)])
                
                if new_window_cost < current_window_cost - 1e-9:
                    for idx, fac_idx in enumerate(fac_indices):
                        p[fac_idx] = loc_indices[new_window_perm[idx]]
                    cost += new_window_cost - current_window_cost
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

    # Initial solution 3: Hungarian-based
    B_p_init = b  # identity mapping initially
    M1 = a @ b.T
    M2 = a.T @ b
    C_init = M1 + M2
    row_ind, col_ind = linear_sum_assignment(C_init)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind

    # Choose the best of the three initial solutions
    cost_rank = compute_cost(p_rank)
    cost_rand = compute_cost(p_rand)
    cost_hung = compute_cost(p_hung)
    
    best_init_cost = min(cost_rank, cost_rand, cost_hung)
    if best_init_cost == cost_rank:
        current_p = p_rank.copy()
        current_cost = cost_rank
        other_p1 = p_rand.copy()
        other_cost1 = cost_rand
        other_p2 = p_hung.copy()
        other_cost2 = cost_hung
    elif best_init_cost == cost_rand:
        current_p = p_rand.copy()
        current_cost = cost_rand
        other_p1 = p_rank.copy()
        other_cost1 = cost_rank
        other_p2 = p_hung.copy()
        other_cost2 = cost_hung
    else:
        current_p = p_hung.copy()
        current_cost = cost_hung
        other_p1 = p_rank.copy()
        other_cost1 = cost_rank
        other_p2 = p_rand.copy()
        other_cost2 = cost_rand

    # First local search
    current_p, current_cost = vectorized_local_search(current_p, current_cost)
    best_p = current_p.copy()
    best_cost = current_cost

    # Try local search from the other initial solutions
    other_p1, other_cost1 = vectorized_local_search(other_p1, other_cost1)
    if other_cost1 < best_cost:
        best_p = other_p1.copy()
        best_cost = other_cost1

    other_p2, other_cost2 = vectorized_local_search(other_p2, other_cost2)
    if other_cost2 < best_cost:
        best_p = other_p2.copy()
        best_cost = other_cost2

    # Iterated Local Search with periodic Hungarian re-optimization and block realignment
    iteration = 0
    while time.time() < deadline - 25:  # reserve 25s for final polish
        iteration += 1

        # Adaptive perturbation strength
        if iteration < 10:
            # Gentle perturbations early
            if iteration % 3 == 0:
                i, j = np.random.choice(n, 2, replace=False)
                p = best_p.copy()
                p[i], p[j] = p[j], p[i]
            else:
                i, j = np.random.choice(n, 2, replace=False)
                p = best_p.copy()
                p[i], p[j] = p[j], p[i]
        elif iteration < 30:
            # Medium perturbations
            if iteration % 5 == 0:
                i = np.random.randint(0, n - 1)
                j = np.random.randint(i + 2, min(i + 15, n))
                p = best_p.copy()
                p[i:j+1] = p[i:j+1][::-1]
            else:
                n_swaps = np.random.randint(2, 6)
                p = best_p.copy()
                for _ in range(n_swaps):
                    i, j = np.random.choice(n, 2, replace=False)
                    p[i], p[j] = p[j], p[i]
        else:
            # Stronger perturbations to escape deep local optima
            if iteration % 7 == 0:
                i = np.random.randint(0, n - 1)
                j = np.random.randint(i + 2, n)
                p = best_p.copy()
                p[i:j+1] = p[i:j+1][::-1]
            elif iteration % 7 == 1:
                n_swaps = np.random.randint(5, 15)
                p = best_p.copy()
                for _ in range(n_swaps):
                    i, j = np.random.choice(n, 2, replace=False)
                    p[i], p[j] = p[j], p[i]
            else:
                n_swaps = np.random.randint(3, 8)
                p = best_p.copy()
                for _ in range(n_swaps):
                    i, j = np.random.choice(n, 2, replace=False)
                    p[i], p[j] = p[j], p[i]

        # Local search from perturbed
        p, cost = vectorized_local_search(p, compute_cost(p))

        # Periodically apply block realignment
        if iteration % 4 == 0 and time.time() < deadline - 35:
            p, cost = block_realignment(p, cost)

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 35:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = vectorized_local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    # Final high-precision polish pass with more iterations
    if time.time() < deadline - 2:
        final_p, final_cost = vectorized_local_search(best_p.copy(), best_cost, max_passes=200)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost

    return best_p.tolist()
