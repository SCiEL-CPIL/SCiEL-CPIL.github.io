"""
Simulated Annealing solver for QAPLIB wil100.
Combines multi-start initialization with highly optimized steepest-descent
local search using vectorized delta matrix computation, followed by a final
SA polish. Includes a structural change: a 2-opt style neighborhood search
that considers reordering of a contiguous block, which can escape local
minima that 1-swap local search cannot.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

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

def _hungarian_initial(a, b):
    """
    Generate an initial permutation using the Hungarian algorithm on a
    linearization of the QAP cost.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Cost[i, j] approximates the marginal cost of assigning facility i to location j
    # based on the average distance and flow.
    cost = np.outer(a_row_sums, b_row_sums)
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] Bp[k, j]
    M1 = a.T @ Bp
    # M2[i, j] = sum_k a[i, k] Bp[k, j]
    M2 = a @ Bp
    
    # Diagonals
    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_Bp = np.diag(Bp)
    
    # Raw terms (sum over all k)
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

def _local_search_fast(a, b, p, time_limit=None, start_time=None):
    """
    Perform steepest descent local search using the vectorized delta matrix.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
        
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        best_delta, best_idx = np.min(D), np.unravel_index(np.argmin(D), D.shape)
        
        if best_delta >= 0:
            break
            
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)
        
    return p, current_cost

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
    
    return T1 + T2 + T3

def _block_move_delta(a, b, p, start, length):
    """
    Compute the cost delta for rotating a block of length 'length' starting at 'start'.
    The block p[start:start+length] becomes p[start+1:start+length] + p[start].
    This is a structural move that can escape 1-swap local minima.
    """
    n = len(p)
    if length < 2 or start + length > n:
        return 0
    
    old_block = p[start:start+length]
    new_block = np.roll(old_block, -1)
    
    # Compute cost contribution of the block in the old and new configuration
    # This is an O(n * length^2) computation which is acceptable for small length
    
    # For simplicity, we compute the full cost difference by considering
    # all interactions between the block elements and the rest
    
    # This is complex to vectorize efficiently for arbitrary length.
    # For now, we implement a simple O(n^2) check for small blocks.
    p_new = p.copy()
    p_new[start:start+length] = new_block
    
    # Compute cost difference
    # This is expensive, so we only do this for small blocks or when needed
    cost_old = _qap_cost(a, b, p)
    cost_new = _qap_cost(a, b, p_new)
    
    return cost_new - cost_old

def _block_local_search(a, b, p, time_limit=None, start_time=None):
    """
    Perform local search using block moves (rotations of small blocks).
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
        
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_length = 3  # Try blocks of length 3
    
    while time.time() - start_time < time_limit:
        improved = False
        for start in range(n - best_length + 1):
            delta = _block_move_delta(a, b, p, start, best_length)
            if delta < 0:
                old_block = p[start:start+best_length]
                p[start:start+best_length] = np.roll(old_block, -1)
                current_cost += int(delta)
                improved = True
                break  # Restart search from beginning
        
        if not improved:
            break
            
    return p, current_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Hungarian initialization (better than greedy)
    p = _hungarian_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine
    p, current_cost = _local_search_fast(a, b, p, time_limit=total_deadline - 20, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # If we still have time, try block local search on the best solution
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        # Try block moves
        p, current_cost = _block_local_search(a, b, p, time_limit=total_deadline - 15, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10:
        # Try a few random starts
        num_random_starts = min(2, int(remaining_time // 15))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 10:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # Local search to refine
            p, current_cost = _local_search_fast(a, b, p, time_limit=total_deadline - 10, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Final polish with SA on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        T_start = max(1.0, best_cost * 0.01)
        T_end = 0.01  # Tighter cooling for more precise local optimization
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
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            iters += 1
    
    if best_p is None:
        best_p = _hungarian_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
