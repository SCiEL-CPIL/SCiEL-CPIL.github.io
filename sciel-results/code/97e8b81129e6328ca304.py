"""Combined solver for QAPLIB wil100: vectorized steepest-descent local search + SA polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]

    # M1[i, j] = sum_k a[k, i] * Bp[k, j]
    M1 = a.T @ Bp
    # M2[i, j] = sum_k a[i, k] * Bp[k, j]
    M2 = a @ Bp

    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_Bp = np.diag(Bp)
    a_diag = np.diag(a)

    # Raw terms (sum over all k)
    T1_raw = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    T2_raw = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T

    # Corrections for k=i and k=j
    C1_part1 = (a_diag[:, None] - a) * (Bp - diag_Bp[:, None])
    C1_part2 = (a.T - a_diag[None, :]) * (diag_Bp[None, :] - Bp)
    C1 = C1_part1 + C1_part2

    C2_part1 = (a_diag[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C2_part2 = (a - a_diag[None, :]) * (diag_Bp[None, :] - Bp.T)
    C2 = C2_part1 + C2_part2

    T3 = (a - a.T) * (Bp.T - Bp)

    D = (T1_raw - C1) + (T2_raw - C2) + T3
    np.fill_diagonal(D, 0)

    return D


def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    """
    pi = p[i]
    pj = p[j]

    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)

    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j

    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)

    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j

    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])

    return T1 + T2 + T3


def _local_search_fast(a, b, p, deadline):
    """
    Perform steepest descent local search using the vectorized delta matrix.
    Returns (p, current_cost).
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)

    while time.time() < deadline:
        D = _compute_delta_matrix(a, b, p)
        best_delta = np.min(D)

        if best_delta >= 0:
            break

        best_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)

    return p, current_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def compute_cost(p):
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    # Initial solution: Hungarian algorithm on linear surrogate
    flow_cent = a.sum(axis=1)
    dist_cent = b.sum(axis=1)
    surrogate_cost = np.outer(flow_cent, dist_cent)
    row_ind, col_ind = linear_sum_assignment(surrogate_cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    best_p = p.copy()
    best_cost = compute_cost(p)

    # Phase 1: Steepest-descent local search from Hungarian start
    p, cur_cost = _local_search_fast(a, b, p.copy(), deadline - 15.0)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_p = p.copy()

    # Phase 2: Try a few random starts with local search
    np.random.seed(42)
    remaining = deadline - time.time()
    num_random_starts = min(3, int(remaining // 25))
    for _ in range(num_random_starts):
        if time.time() > deadline - 15.0:
            break
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, cost_rand = _local_search_fast(a, b, p_rand, deadline - 15.0)
        if cost_rand < best_cost:
            best_cost = cost_rand
            best_p = p_rand.copy()

    # Phase 3: SA polish on the best solution
    remaining = deadline - time.time()
    if remaining > 5 and best_p is not None:
        p = best_p.copy()
        cur_cost = best_cost

        T_start = max(best_cost * 0.01, 1000.0)
        T_end = max(best_cost * 0.0001, 1.0)
        total_sa_time = min(remaining - 5.0, 100.0)
        sa_start = time.time()

        while time.time() < deadline - 3.0:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue

            delta = _compute_swap_delta(a, b, p, i, j)

            elapsed = time.time() - sa_start
            T = T_start * (T_end / T_start) ** (elapsed / total_sa_time)
            T = max(T, T_end)

            if delta < 0:
                p[i], p[j] = p[j], p[i]
                cur_cost += int(delta)
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_p = p.copy()
            else:
                if np.random.random() < np.exp(-delta / T):
                    p[i], p[j] = p[j], p[i]
                    cur_cost += int(delta)

    # Phase 4: Final steepest-descent polish on best solution
    if time.time() < deadline - 2.0:
        p_final = best_p.copy()
        p_final, cost_final = _local_search_fast(a, b, p_final, deadline - 1.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
