"""
Simulated Annealing solver for QAPLIB wil100.
Combines multi-start initialization with incremental steepest-descent local
search (O(n) per swap instead of O(n^2) matrix recomputation) and a final
SA polish using vectorized delta updates.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_initial(a, b):
    """
    Generate an initial permutation using the Hungarian algorithm on a
    linearization of the QAP cost.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    cost = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    M1 = a.T @ Bp
    M2 = a @ Bp
    
    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_Bp = np.diag(Bp)
    
    T1_raw = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    T2_raw = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T
    
    a_diag = np.diag(a)
    
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
    Returns the integer delta.
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
    
    return int(T1 + T2 + T3)

def _local_search_incremental(a, b, p, time_limit, start_time):
    """
    Steepest descent local search with incremental delta matrix updates.
    After each swap of positions i,j, only rows i and j of D need updating.
    This reduces per-iteration cost from O(n^2) matrix ops to O(n) vector ops.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    D = _compute_delta_matrix(a, b, p)
    
    # Precompute helper arrays for fast row updates
    Bp = b[np.ix_(p, p)]
    
    while time.time() - start_time < time_limit:
        best_delta = np.min(D)
        if best_delta >= 0:
            break
        
        best_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = best_idx
        pi, pj = p[i], p[j]
        
        # Perform the swap
        p[i], p[j] = pj, pi
        Bp[i], Bp[j] = Bp[j], Bp[i]
        current_cost += int(best_delta)
        
        # Update only rows i and j of D
        # D[i, k] for all k != i, j changes because Bp[i] changed
        # D[j, k] for all k != i, j changes because Bp[j] changed
        # D[k, i] and D[k, j] also change
        
        # Recompute D[i, :] for all k
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            # Compute delta for swapping positions i and k
            # Using the same formula as _compute_swap_delta but inline
            diff_a_col = a[:, i] - a[:, k]
            diff_b_row = b[p, p[i]] - b[p, pk]
            T1 = np.dot(diff_a_col, diff_b_row)
            T1 -= (a[i, i] - a[i, k]) * (b[p[i], p[i]] - b[p[i], pk])
            T1 -= (a[k, i] - a[k, k]) * (b[pk, pk] - b[pk, p[i]])
            
            diff_a_row = a[i, :] - a[k, :]
            diff_b_col = b[p[i], p] - b[pk, p]
            T2 = np.dot(diff_a_row, diff_b_col)
            T2 -= (a[i, i] - a[k, i]) * (b[p[i], p[i]] - b[p[i], p[i]])
            T2 -= (a[i, k] - a[k, k]) * (b[p[i], pk] - b[p[i], p[i]])
            
            T3 = (a[i, k] - a[k, i]) * (b[p[i], pk] - b[pk, p[i]])
            D[i, k] = int(T1 + T2 + T3)
            D[k, i] = D[i, k]
        
        # Recompute D[j, :] for all k
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            diff_a_col = a[:, j] - a[:, k]
            diff_b_row = b[p, p[j]] - b[p, pk]
            T1 = np.dot(diff_a_col, diff_b_row)
            T1 -= (a[j, j] - a[j, k]) * (b[p[j], p[j]] - b[p[j], pk])
            T1 -= (a[k, j] - a[k, k]) * (b[pk, pk] - b[pk, p[j]])
            
            diff_a_row = a[j, :] - a[k, :]
            diff_b_col = b[p[j], p] - b[pk, p]
            T2 = np.dot(diff_a_row, diff_b_col)
            T2 -= (a[j, j] - a[k, j]) * (b[p[j], p[j]] - b[p[j], p[j]])
            T2 -= (a[j, k] - a[k, k]) * (b[p[j], pk] - b[p[j], p[j]])
            
            T3 = (a[j, k] - a[k, j]) * (b[p[j], pk] - b[pk, p[j]])
            D[j, k] = int(T1 + T2 + T3)
            D[k, j] = D[j, k]
        
        # Update D[i, j] and D[j, i]
        D[i, j] = _compute_swap_delta(a, b, p, i, j)
        D[j, i] = D[i, j]
        
        np.fill_diagonal(D, 0)
    
    return p, current_cost

def _final_polish(a, b, p, current_cost, time_limit, start_time):
    """
    Final SA polish using vectorized swap deltas.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = current_cost
    
    T_start = max(1.0, current_cost * 0.005)
    T_end = 0.01
    elapsed_total = time_limit
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = current_cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / elapsed_total)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Hungarian initialization
    p = _hungarian_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    p, current_cost = _local_search_incremental(a, b, p, time_limit=total_deadline - 15, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Start 2: Greedy initialization
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    p, current_cost = _local_search_incremental(a, b, p, time_limit=total_deadline - 15, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Start 3: Random initialization
    p = _random_initial(n)
    current_cost = _qap_cost(a, b, p)
    p, current_cost = _local_search_incremental(a, b, p, time_limit=total_deadline - 15, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Additional random starts if time permits
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15:
        for _ in range(3):
            if time.time() - start_time > total_deadline - 15:
                break
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            p, current_cost = _local_search_incremental(a, b, p, time_limit=total_deadline - 15, start_time=start_time)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Final polish with SA on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 3 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        p, current_cost = _final_polish(a, b, p, current_cost, remaining_time - 1, start_time)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
