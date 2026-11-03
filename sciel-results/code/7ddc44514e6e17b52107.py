"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    """
    Solve the Quadratic Assignment Problem for a n x n instance.
    Uses rank-matching initialization followed by efficient 2-opt local search
    with time-bounded iterated local search and adaptive perturbations.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin within 120s limit

    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def compute_cost(p):
        """Compute the exact QAP cost for assignment p."""
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))

    def greedy_init():
        """Rank-matching: high-flow facilities to low-distance locations."""
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def swap_delta_fast(p, i, j):
        """
        Fast O(n) swap delta:
        delta = (a[i,:]-a[j,:]) @ (b[pj]-b[pi]) + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])
        """
        pi = p[i]
        pj = p[j]
        row_diff = b[pj] - b[pi]
        col_diff = b[:, pj] - b[:, pi]
        delta = np.dot(a[i, :] - a[j, :], row_diff) + np.dot(a[:, i] - a[:, j], col_diff)
        return delta

    def local_search(p, max_passes=100):
        """2-opt local search with first-improvement and random offset."""
        p = p.copy()
        cost = compute_cost(p)
        n_local = len(p)

        for _pass in range(max_passes):
            if time.time() > deadline:
                break

            improved = False
            offset = np.random.randint(n_local)
            for ii in range(n_local):
                i = (offset + ii) % n_local
                for jj in range(ii + 1, n_local):
                    j = (offset + jj) % n_local
                    if j == i:
                        continue
                    delta = swap_delta_fast(p, i, j)
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if time.time() > deadline:
                    return p, cost

            if not improved:
                break

        return p, cost

    def perturb(p, num_swaps):
        """Apply random multi-swap perturbation."""
        p = p.copy()
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        return p

    # Initial solution via rank matching
    best_p = greedy_init()
    best_cost = compute_cost(best_p)

    # First local search to convergence
    best_p, best_cost = local_search(best_p, max_passes=200)

    def hungarian_reassign(p, cost_matrix=None):
        """
        Use Hungarian algorithm to find a globally optimal reassignment
        based on marginal contributions from current assignment.
        """
        from scipy.optimize import linear_sum_assignment
        if cost_matrix is None:
            # Build effective cost matrix from current assignment
            # c[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
            # This is the marginal cost of placing facility i at location j
            n_local = len(p)
            # Precompute for efficiency
            c = np.zeros((n_local, n_local), dtype=np.float64)
            for i in range(n_local):
                # Row contribution: sum_k a[i,k] * b[j, p[k]]
                # = sum_k a[i,k] * b[j, p[k]]
                # Vectorized: for each j, sum over k of a[i,k] * b[j, p[k]]
                # = b[:, p] @ a[i, :]  (since b[j, p[k]] for all k, then dot with a[i,:])
                c[i, :] = (b[:, p] @ a[i, :])
            # Column contribution: sum_k a[k, i] * b[p[k], j]
            # = a[:, i] @ b[p, :]
            c += (a[:, :].T @ b[p, :])
        
        row_ind, col_ind = linear_sum_assignment(c)
        p_new = np.empty(n_local, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new

    # Iterated local search with adaptive perturbation
    stall_count = 0
    stall_threshold = 4
    use_hungarian = False

    while time.time() < deadline - 5.0:
        if stall_count < stall_threshold:
            num_swaps = 2
            use_hungarian = False
        elif stall_count < 2 * stall_threshold:
            num_swaps = 4
            use_hungarian = False
        elif stall_count < 4 * stall_threshold:
            num_swaps = 6
            use_hungarian = False
        else:
            use_hungarian = True

        if use_hungarian:
            p_pert = hungarian_reassign(best_p)
        else:
            p_pert = perturb(best_p, num_swaps)
        
        p_new, cost_new = local_search(p_pert, max_passes=80)

        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
            stall_count = 0
        else:
            if stall_count > 3 * stall_threshold and np.random.random() < 0.15:
                best_p = p_new
                best_cost = cost_new
            stall_count += 1

        # If heavily stalled, try fresh initialization
        if stall_count > 12:
            p_fresh = greedy_init()
            p_new, cost_new = local_search(p_fresh, max_passes=80)
            if cost_new < best_cost:
                best_p = p_new
                best_cost = cost_new
            stall_count = 0

    # Final polish with multiple passes
    if time.time() < deadline - 2.0:
        best_p, best_cost = local_search(best_p, max_passes=50)
        # One more polish pass
        if time.time() < deadline - 1.0:
            best_p, best_cost = local_search(best_p, max_passes=20)

    return best_p.tolist()

# EVOLVE-BLOCK-END
