"""
QAP solver for wil100 using multi-start local search with 3-opt cyclic
exchange neighborhood and SA polish.
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

def _compute_2swap_delta(a, b, p):
    """
    Compute the full n x n matrix of 2-swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
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
    
    # Corrections for k=i and k=j
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

def _compute_3cycle_deltas(a, b, p, i, j, k):
    """
    Compute the cost delta for a 3-cycle: positions i->j->k->i.
    That is, facility at i goes to j, facility at j goes to k, facility at k goes to i.
    """
    pi = p[i]
    pj = p[j]
    pk = p[k]
    
    # The 3-cycle changes: p[i]=pi->pj, p[j]=pj->pk, p[k]=pk->pi
    # Delta = sum over all (a,b) pairs affected
    
    # Affected terms are those involving positions i, j, k or facilities pi, pj, pk
    # Let's compute the delta directly
    
    # For each pair (x, y) in {i,j,k} x {i,j,k}, the contribution changes
    # Old: a[x,y] * b[p[x], p[y]]
    # New: a[x,y] * b[p_new[x], p_new[y]]
    
    # p_new: p_new[i] = pj, p_new[j] = pk, p_new[k] = pi
    
    delta = 0
    for x in [i, j, k]:
        for y in [i, j, k]:
            if x == i and y == i:
                old_b = b[pi, pi]
                new_b = b[pj, pj]
            elif x == i and y == j:
                old_b = b[pi, pj]
                new_b = b[pj, pk]
            elif x == i and y == k:
                old_b = b[pi, pk]
                new_b = b[pj, pi]
            elif x == j and y == i:
                old_b = b[pj, pi]
                new_b = b[pk, pj]
            elif x == j and y == j:
                old_b = b[pj, pj]
                new_b = b[pk, pk]
            elif x == j and y == k:
                old_b = b[pj, pk]
                new_b = b[pk, pi]
            elif x == k and y == i:
                old_b = b[pk, pi]
                new_b = b[pi, pj]
            elif x == k and y == j:
                old_b = b[pk, pj]
                new_b = b[pi, pk]
            elif x == k and y == k:
                old_b = b[pk, pk]
                new_b = b[pi, pi]
            else:
                continue
            delta += a[x, y] * (new_b - old_b)
    
    return delta

def _local_search_2swap(a, b, p, time_limit, start_time):
    """
    Perform steepest descent local search using 2-swap moves.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    while time.time() - start_time < time_limit:
        D = _compute_2swap_delta(a, b, p)
        best_delta, best_idx = np.min(D), np.unravel_index(np.argmin(D), D.shape)
        
        if best_delta >= 0:
            break
            
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)
        
    return p, current_cost

def _local_search_3cycle(a, b, p, time_limit, start_time, max_3opt_iter=50):
    """
    Perform local search using 3-cycle moves to escape 2-opt local optima.
    We try random 3-cycles and accept if improving.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    no_improve_count = 0
    
    while time.time() - start_time < time_limit and no_improve_count < max_3opt_iter:
        i = np.random.randint(n)
        j = np.random.randint(n)
        k = np.random.randint(n)
        
        if i == j or i == k or j == k:
            continue
        
        delta = _compute_3cycle_deltas(a, b, p, i, j, k)
        
        if delta < 0:
            # Apply the 3-cycle: i->j->k->i
            pi = p[i]
            pj = p[j]
            pk = p[k]
            p[i] = pj
            p[j] = pk
            p[k] = pi
            current_cost += int(delta)
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return p, current_cost

def _sa_polish(a, b, p, current_cost, time_limit, start_time):
    """
    Simulated annealing polish phase.
    """
    n = len(p)
    T_start = max(1.0, current_cost * 0.01)
    T_end = 0.1
    
    best_cost = current_cost
    best_p = p.copy()
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        # Use 2-swap delta computation for SA moves
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
        
        delta = T1 + T2 + T3
        new_cost = current_cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / time_limit)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 145.0
    start_time = time.time()
    
    best_p = None
    best_cost = float('inf')
    
    # Multi-start strategy
    starts = [
        ("greedy", _greedy_initial(a, b)),
        ("random1", _random_initial(n)),
        ("random2", _random_initial(n)),
        ("random3", _random_initial(n)),
    ]
    
    for name, p in starts:
        if time.time() - start_time > total_deadline - 30:
            break
        
        current_cost = _qap_cost(a, b, p)
        
        # Phase 1: 2-swap local search
        time_left = total_deadline - (time.time() - start_time) - 20
        if time_left > 5:
            p, current_cost = _local_search_2swap(a, b, p, time_left, start_time)
        
        # Phase 2: 3-cycle local search to escape 2-opt optima
        time_left = total_deadline - (time.time() - start_time) - 10
        if time_left > 3:
            p, current_cost = _local_search_3cycle(a, b, p, time_left, start_time)
        
        # Phase 3: SA polish
        time_left = total_deadline - (time.time() - start_time) - 2
        if time_left > 2:
            p, current_cost = _sa_polish(a, b, p, current_cost, time_left, start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
