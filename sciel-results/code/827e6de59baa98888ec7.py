"""
Simulated Annealing solver for QAPLIB wil100.
Combines multi-start greedy+random initialization with a highly optimized
steepest-descent local search using vectorized delta matrix computation,
followed by SA/tabu polish phases.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns the integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    # Term 1: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract contributions from k=i and k=j
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # Term 2: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract contributions from k=i and k=j
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # Term 3: (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] * Bp[k, j] = (a.T @ Bp)[i, j]
    M1 = a.T @ Bp  # shape (n, n)
    
    # M2[i, j] = sum_k a[i, k] * Bp[j, k] = (a @ Bp.T)[i, j]
    M2 = a @ Bp.T  # shape (n, n)
    
    # For swap (i, j), the delta is:
    # T1 = sum_{k!=i,j} (a[k,i]-a[k,j])*(Bp[k,j]-Bp[k,i])
    #    = sum_k a[k,i]*Bp[k,j] - sum_k a[k,i]*Bp[k,i] - sum_k a[k,j]*Bp[k,j] + sum_k a[k,j]*Bp[k,i]
    #    = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    #    minus corrections for k=i and k=j
    
    # T2 = sum_{k!=i,j} (a[i,k]-a[j,k])*(Bp[j,k]-Bp[i,k])
    #    = sum_k a[i,k]*Bp[j,k] - sum_k a[i,k]*Bp[i,k] - sum_k a[j,k]*Bp[j,k] + sum_k a[j,k]*Bp[i,k]
    #    = M2[i,j] - M2[i,i] - M2[j,j] + M2[j,i]
    #    minus corrections for k=i and k=j
    
    # T3 = (a[i,j]-a[j,i])*(Bp[j,i]-Bp[i,j])
    
    # Let's compute the full delta matrix element by element using broadcasting.
    # For all pairs (i, j):
    # T1[i,j] = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    # T2[i,j] = M2[i,j] - M2[i,i] - M2[j,j] + M2[j,i]
    # T3[i,j] = (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    
    diag_M1 = np.diag(M1)  # M1[i,i]
    diag_M2 = np.diag(M2)  # M2[i,i]
    
    # T1 for all (i,j):
    T1_mat = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    
    # T2 for all (i,j):
    T2_mat = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T
    
    # T3 for all (i,j):
    T3_mat = (a - a.T) * (Bp.T - Bp)
    
    # Now subtract the k=i and k=j corrections.
    # Correction for T1, k=i: (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    # Correction for T1, k=j: (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    corr_T1_i = (a[np.diag_indices(n)] - a) * (Bp - Bp[np.diag_indices(n)][:, None])
    # Wait, let me be more careful.
    # corr_T1_i[i,j] = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    # a[i,i] is diag_a[i], a[i,j] is a[i,j]
    # Bp[i,j] is Bp[i,j], Bp[i,i] is diag_Bp[i]
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    corr_T1_i = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    
    # corr_T1_j[i,j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    # = (a.T[i,j] - diag_a[j]) * (diag_Bp[j] - Bp[j,i])
    corr_T1_j = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T1_mat = T1_mat - corr_T1_i - corr_T1_j
    
    # Corrections for T2:
    # corr_T2_i[i,j] = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
    # = (diag_a[i] - a[j,i]) * (Bp[j,i] - diag_Bp[i])
    # = (diag_a[i] - a.T[i,j]) * (Bp.T[i,j] - diag_Bp[i])
    corr_T2_i = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    
    # corr_T2_j[i,j] = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
    # = (a[i,j] - diag_a[j]) * (diag_Bp[j] - Bp[i,j])
    corr_T2_j = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T2_mat = T2_mat - corr_T2_i - corr_T2_j
    
    D = T1_mat + T2_mat + T3_mat
    
    # Set diagonal to 0 (no swap with self)
    np.fill_diagonal(D, 0)
    
    return D

def _local_search_fast(a, b, p, max_iter=None, time_limit=None, start_time=None):
    """
    Perform steepest descent local search using the full delta matrix.
    Each iteration computes the full n x n delta matrix in O(n^3) via matmul,
    then finds the best improving swap.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 200000
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    improved = True
    while improved and max_iter > 0:
        if time.time() - start_time > time_limit:
            break
        
        D = _compute_delta_matrix(a, b, p)
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Find the most negative entry (best improving move)
        # Only consider upper triangle to avoid duplicates
        upper = np.triu_indices(n, k=1)
        D_upper = D[upper]
        min_idx = np.argmin(D_upper)
        min_val = D_upper[min_idx]
        
        if min_val < 0:
            # Convert linear index back to (i, j)
            # upper[0] and upper[1] contain the row and col indices
            best_i = upper[0][min_idx]
            best_j = upper[1][min_idx]
            best_delta = min_val
            
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
            max_iter -= 1
        else:
            improved = False
    
    return p, current_cost

def _tabu_search(a, b, p, max_iter=None, time_limit=None, start_time=None, tabu_len=20):
    """
    Tabu search to escape local minima.
    Maintains a tabu list of recently used swaps.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 50000
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    # Tabu list: store (i, j) pairs with expiry iteration
    tabu = {}  # (min(i,j), max(i,j)) -> iteration when it becomes non-tabu
    
    best_cost = current_cost
    best_p = p.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        D = _compute_delta_matrix(a, b, p)
        
        # Find best non-tabu move, or best move if all are tabu (aspiration)
        upper = np.triu_indices(n, k=1)
        D_upper = D[upper]
        
        # Create a mask for non-tabu moves
        mask = np.ones(len(D_upper), dtype=bool)
        for idx in range(len(D_upper)):
            i = upper[0][idx]
            j = upper[1][idx]
            key = (min(i, j), max(i, j))
            if key in tabu and tabu[key] > it:
                mask[idx] = False
        
        if not np.any(mask):
            # All moves are tabu; use aspiration (allow the best move)
            mask = np.ones(len(D_upper), dtype=bool)
        
        # Find best among non-tabu
        D_masked = np.where(mask, D_upper, np.inf)
        min_idx = np.argmin(D_masked)
        min_val = D_masked[min_idx]
        
        if min_val == np.inf:
            break
        
        best_i = upper[0][min_idx]
        best_j = upper[1][min_idx]
        best_delta = min_val
        
        # Aspiration: allow tabu move if it improves the best known cost
        key = (min(best_i, best_j), max(best_i, best_j))
        if key in tabu and tabu[key] > it and current_cost + best_delta > best_cost:
            # Don't take this tabu move unless it improves best
            # Find next best non-tabu
            D_masked[min_idx] = np.inf
            min_idx2 = np.argmin(D_masked)
            if D_masked[min_idx2] == np.inf:
                break
            best_i = upper[0][min_idx2]
            best_j = upper[1][min_idx2]
            best_delta = D_masked[min_idx2]
            key = (min(best_i, best_j), max(best_i, best_j))
        
        # Perform the swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update tabu list
        tabu[key] = it + tabu_len
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p, best_cost

def _local_search(a, b, p, max_iter=None, time_limit=None, start_time=None):
    """
    Perform steepest descent local search until no improving move is found
    or limits are reached.
    Returns the improved permutation and cost.
    Uses the fast delta matrix computation.
    """
    return _local_search_fast(a, b, p, max_iter=max_iter, time_limit=time_limit, start_time=start_time)

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Multi-start: try several different initializations
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Greedy initialization
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine (fast version)
    p, current_cost = _local_search_fast(a, b, p, max_iter=100000, time_limit=total_deadline - 30, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15:
        # Try random starts with local search
        num_random_starts = min(4, max(1, int(remaining_time // 12)))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 25:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # Local search to refine
            p, current_cost = _local_search_fast(a, b, p, max_iter=50000, time_limit=total_deadline - 25, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Phase 2: Tabu search to escape local minima
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        p_tabu, cost_tabu = _tabu_search(a, b, p, max_iter=50000, 
                                          time_limit=total_deadline - 10, 
                                          start_time=start_time, tabu_len=20)
        
        if cost_tabu < best_cost:
            best_cost = cost_tabu
            best_p = p_tabu.copy()
    
    # Phase 3: Final SA polish
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 3 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        T_start = max(1.0, best_cost * 0.005)
        T_end = 0.05
        elapsed_total = total_deadline
        iters = 0
        
        while time.time() - start_time < total_deadline - 1:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _compute_swap_delta(a, b, p, i, j)
            new_cost = current_cost + delta
            
            elapsed = time.time() - start_time
            T = T_start * (T_end / T_start) ** (elapsed / elapsed_total)
            
            if delta < 0 or (T > 1e-10 and np.random.rand() < np.exp(-delta / T)):
                p[i], p[j] = p[j], p[i]
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            iters += 1
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
