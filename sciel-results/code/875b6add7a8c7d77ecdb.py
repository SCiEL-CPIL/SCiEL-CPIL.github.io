"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


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
        Correct O(n) swap delta for QAP.
        delta = (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:])[p] + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])[p]
        """
        pi = p[i]
        pj = p[j]
        diff_a_row = a[i, :] - a[j, :]
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[pj, :] - b[pi, :]
        diff_b_col = b[:, pj] - b[:, pi]
        delta = np.dot(diff_a_row, diff_b_row[p]) + np.dot(diff_a_col, diff_b_col[p])
        return delta

    def local_search(p, max_passes=100, time_limit=None):
        """2-opt local search with best-improvement strategy."""
        if time_limit is None:
            time_limit = deadline
        p = p.copy()
        cost = compute_cost(p)
        n_local = len(p)

        for _pass in range(max_passes):
            if time.time() > time_limit:
                break

            improved = False
            best_delta = 0.0
            best_i, best_j = -1, -1

            for i in range(n_local):
                for j in range(i + 1, n_local):
                    delta = swap_delta_fast(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
                if time.time() > time_limit:
                    return p, cost

            if best_i == -1:
                break

            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            improved = True

        return p, cost

    def hungarian_reassign(p, time_limit=None):
        """
        Use Hungarian algorithm to find a globally optimal reassignment
        based on marginal costs from the current assignment.
        """
        if time_limit is None:
            time_limit = deadline
        n_local = len(p)

        # Compute marginal cost matrix: c[i,j] = cost contribution of placing facility i at location j
        # given the current assignment of all other facilities
        C = np.zeros((n_local, n_local), dtype=np.float64)
        for i in range(n_local):
            if time.time() > time_limit:
                break
            C[i, :] = a[i, :] @ b
            C[i, :] += a[:, i] @ b.T

        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n_local, dtype=np.int64)
        new_p[row_ind] = col_ind

        new_cost = compute_cost(new_p)
        return new_p, new_cost

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

    # Iterated local search with adaptive perturbation and Hungarian reassignment
    stall_count = 0
    stall_threshold = 4
    hungarian_count = 0

    while time.time() < deadline - 5.0:
        # Try Hungarian reassignment periodically to escape local optima
        if hungarian_count % 5 == 0 and time.time() < deadline - 10.0:
            p_hung, cost_hung = hungarian_reassign(best_p)
            if cost_hung < best_cost:
                best_p = p_hung
                best_cost = cost_hung
                stall_count = 0
                best_p, best_cost = local_search(best_p, max_passes=50)
            hungarian_count += 1

        if stall_count < stall_threshold:
            num_swaps = 2
        elif stall_count < 2 * stall_threshold:
            num_swaps = 4
        else:
            num_swaps = 6

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

    # Final polish
    if time.time() < deadline - 2.0:
        best_p, best_cost = local_search(best_p, max_passes=30)

    return best_p.tolist()

# EVOLVE-BLOCK-END
