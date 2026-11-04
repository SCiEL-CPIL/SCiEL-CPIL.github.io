import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Vectorized computation of all swap deltas for permutation p.
    Returns a 2D array of shape (n, n) where entry [i,j] is the delta for swapping i and j.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i,j] = sum_k a[k,i] * b[p[k], p[j]] = (a.T @ Bp)[i,j]
    M1 = a.T @ Bp
    # M3[i,j] = sum_k a[i,k] * b[p[j], p[k]] = (a @ Bp.T)[i,j]
    M3 = a @ Bp.T
    
    # t1_raw[i,j] = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    t1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    # t2_raw[i,j] = M3[i,j] - M3[i,i] - M3[j,j] + M3[j,i]
    t2_raw = M3 - np.diag(M3)[:, None] - np.diag(M3)[None, :] + M3.T
    
    # Corrections
    A_ii = np.diag(a)
    A_ii_col = A_ii[:, None]
    A_ii_row = A_ii[None, :]
    
    Bp_ii = np.diag(Bp)
    Bp_ii_col = Bp_ii[:, None]
    Bp_ii_row = Bp_ii[None, :]
    
    corr1_1 = (A_ii_col - a) * (Bp - Bp_ii_row)
    corr1_2 = (a.T - A_ii_row) * (Bp_ii_col - Bp.T)
    corr2_1 = (A_ii_col - a.T) * (Bp.T - Bp_ii_row)
    corr2_2 = (a - A_ii_row) * (Bp_ii_col - Bp)
    t3 = (a - a.T) * (Bp.T - Bp)
    
    D = t1_raw - corr1_1 - corr1_2 + t2_raw - corr2_1 - corr2_2 + t3
    
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    
    return D

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    O(n) time complexity.
    """
    pi = p[i]
    pj = p[j]
    
    t1 = np.dot(a[:, i] - a[:, j], b[p, pj] - b[p, pi])
    t1 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    t1 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    t2 = np.dot(a[i, :] - a[j, :], b[pj, p] - b[pi, p])
    t2 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    t2 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    t3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return int(t1 + t2 + t3)

def _local_search(a, b, p, current_cost, time_limit=10.0):
    """
    Steepest-descent local search using vectorized delta matrix.
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        best_delta = D[i, j]
        
        if best_delta < 0:
            p[i], p[j] = p[j], p[i]
            cost += int(best_delta)
        else:
            break
            
    return p, cost

def _sa_run(a, b, p_init, current_cost, duration, T_start, T_end):
    """
    Run simulated annealing for a given duration.
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    # Pre-compute some constants? No, matrices are static, but p changes.
    # We can pre-compute row/col sums of a? Not directly helpful for swap delta.
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        new_cost = cost + delta
        
        elapsed = time.time() - start_time
        # Exponential cooling
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
    return best_p, best_cost

def _lsa_initialization(a, b):
    """
    Initialize using Linear Sum Assignment approximation.
    Cost matrix for LSA: C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
    This approximates the QAP cost by ignoring the permutation constraint on the other index.
    """
    n = len(a)
    # C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
    # = (a @ b.T)[i, j] + (a.T @ b)[i, j]
    C = a @ b.T + a.T @ b
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 145.0  # Safety margin for 150s limit
    start_time = time.time()
    
    # Phase 1: Initial Solutions
    # 1. LSA-based initialization
    p_lsa = _lsa_initialization(a, b)
    cost_lsa = _qap_cost(a, b, p_lsa)
    
    # 2. Greedy initialization (fallback or comparison)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[heavy] = central
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    # 3. Random initialization
    p_random = np.random.permutation(n).astype(np.int64)
    cost_random = _qap_cost(a, b, p_random)
    
    # Pick the best initial solution
    candidates = [
        (cost_lsa, p_lsa),
        (cost_greedy, p_greedy),
        (cost_random, p_random)
    ]
    candidates.sort(key=lambda x: x[0])
    best_p = candidates[0][1].copy()
    best_cost = candidates[0][0]
    
    # Phase 2: Initial Local Search Polish
    elapsed = time.time() - start_time
    ls_time = min(10.0, total_deadline - elapsed - 30.0)
    if ls_time > 0:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, time_limit=ls_time)
        
    # Phase 3: Multi-start SA
    elapsed = time.time() - start_time
    sa_time_total = total_deadline - elapsed - 10.0
    if sa_time_total <= 0:
        # If time is tight, just return current best
        return best_p.tolist()
    
    num_sa_runs = 6
    sa_time_each = sa_time_total / num_sa_runs
    
    for k in range(num_sa_runs):
        if time.time() - start_time > total_deadline - 10.0:
            break
            
        if k == 0:
            p_start = best_p.copy()
            c_start = best_cost
        else:
            # Perturb the best solution or start from LSA/random
            if k % 2 == 1:
                p_start = best_p.copy()
                # Apply a few random swaps to perturb
                for _ in range(n // 2):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p_start[i], p_start[j] = p_start[j], p_start[i]
                c_start = _qap_cost(a, b, p_start)
            else:
                p_start = _lsa_initialization(a, b)
                c_start = _qap_cost(a, b, p_start)
            
            # Short local search to polish start point
            p_start, c_start = _local_search(a, b, p_start, c_start, time_limit=2.0)
            
            # SA parameters
            T_start = 1000.0
            T_end = 1.0
            
            p_sa, c_sa = _sa_run(a, b, p_start, c_start, sa_time_each, T_start, T_end)
            
            # Polish the SA result
            p_sa, c_sa = _local_search(a, b, p_sa, c_sa, time_limit=5.0)
            
            if c_sa < best_cost:
                best_cost = c_sa
                best_p = p_sa.copy()
                
    # Phase 4: Final Polish
    elapsed = time.time() - start_time
    final_ls_time = max(0.0, total_deadline - elapsed - 1.0)
    if final_ls_time > 0:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, time_limit=final_ls_time)
        
    return best_p.tolist()
