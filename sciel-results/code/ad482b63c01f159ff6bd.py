"""
Vectorized Parallel Simulated Annealing for QAPLIB wil100.
Strategy:
1. Hungarian-matching initialization (row-sum centrality alignment).
2. Maintain a population of independent SA chains.
3. Each iteration: generate random swap candidates for all chains, compute delta costs in one vectorized batch, accept/reject based on current temperature.
4. Time-based cooling schedule.
5. Return the best permutation found across all chains.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian matching on row-sum centralities.
    Heavy facilities (high row sum in a) go to central locations (low row sum in b).
    """
    n = len(a)
    a_row_sums = a.sum(axis=1).astype(np.float64)
    b_row_sums = b.sum(axis=1).astype(np.float64)
    
    # Cost matrix: we want to minimize sum of a_row_sums[i] * b_row_sums[p[i]]
    # This is a linear assignment problem.
    # Use negative to convert minimization to maximization for linear_sum_assignment
    cost_matrix = -np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _compute_batch_deltas(a, b, p, i_arr, j_arr):
    """
    Compute delta costs for a batch of swaps (i_arr, j_arr) on permutation p.
    Returns an array of delta costs.
    """
    # p: (n,) current permutation
    # i_arr, j_arr: (m,) arrays of swap indices
    
    # For each swap (i, j), the delta is:
    # sum_k (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
    # + sum_k (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    # - corrections for k=i and k=j
    
    # We'll compute this vectorized over the batch dimension.
    # m = len(i_arr)
    # For each swap index m_idx:
    #   i = i_arr[m_idx], j = j_arr[m_idx]
    #   pi = p[i], pj = p[j]
    
    # Gather the needed values
    i_idx = i_arr  # (m,)
    j_idx = j_arr  # (m,)
    
    pi = p[i_idx]  # (m,) facility at position i
    pj = p[j_idx]  # (m,) facility at position j
    
    # diff_a_row[m, k] = a[i, k] - a[j, k]
    diff_a_row = a[i_idx, :] - a[j_idx, :]  # (m, n)
    
    # diff_a_col[m, k] = a[k, i] - a[k, j]
    diff_a_col = a[:, i_idx] - a[:, j_idx]  # (n, m) -> transpose to (m, n)
    diff_a_col = diff_a_col.T  # (m, n)
    
    # diff_b_col[m, k] = b[pj, p[k]] - b[pi, p[k]]
    # b[pj, p[k]]: for each m, b[pj[m], p[k]] for all k
    b_pj_p = b[np.ix_(pj, p)]  # (m, n)
    b_pi_p = b[np.ix_(pi, p)]  # (m, n)
    diff_b_col = b_pj_p - b_pi_p  # (m, n)
    
    # diff_b_row[m, k] = b[p[k], pj] - b[p[k], pi]
    b_p_pj = b[np.ix_(p, pj)]  # (n, m) -> transpose
    b_p_pj = b_p_pj.T  # (m, n)
    b_p_pi = b[np.ix_(p, pi)]  # (n, m) -> transpose
    b_p_pi = b_p_pi.T  # (m, n)
    diff_b_row = b_p_pj - b_p_pi  # (m, n)
    
    # Term 1: sum_k diff_a_row[m,k] * diff_b_col[m,k]
    T1 = np.sum(diff_a_row * diff_b_col, axis=1)  # (m,)
    
    # Term 2: sum_k diff_a_col[m,k] * diff_b_row[m,k]
    T2 = np.sum(diff_a_col * diff_b_row, axis=1)  # (m,)
    
    # Corrections for k=i:
    # T1 correction: (a[i,i]-a[j,i]) * (b[pj,pi]-b[pi,pi])
    corr_i_T1 = (a[i_idx, i_idx] - a[j_idx, i_idx]) * (b[pj, pi] - b[pi, pi])  # (m,)
    
    # T2 correction: (a[i,i]-a[i,j]) * (b[pi,pj]-b[pi,pi])
    corr_i_T2 = (a[i_idx, i_idx] - a[i_idx, j_idx]) * (b[pi, pj] - b[pi, pi])  # (m,)
    
    # Corrections for k=j:
    # T1 correction: (a[i,j]-a[j,j]) * (b[pj,pj]-b[pi,pj])
    corr_j_T1 = (a[i_idx, j_idx] - a[j_idx, j_idx]) * (b[pj, pj] - b[pi, pj])  # (m,)
    
    # T2 correction: (a[j,i]-a[j,j]) * (b[pj,pj]-b[pj,pi])
    corr_j_T2 = (a[j_idx, i_idx] - a[j_idx, j_idx]) * (b[pj, pj] - b[pj, pi])  # (m,)
    
    # Direct terms:
    direct1 = (a[i_idx, j_idx] - a[j_idx, i_idx]) * (b[pj, pi] - b[pi, pj])  # (m,)
    direct2 = a[i_idx, i_idx] * (b[pj, pj] - b[pi, pi])  # (m,)
    direct3 = a[j_idx, j_idx] * (b[pi, pi] - b[pj, pj])  # (m,)
    
    deltas = T1 + T2 - corr_i_T1 - corr_i_T2 - corr_j_T1 - corr_j_T2 + direct1 + direct2 + direct3
    
    return deltas.astype(np.int64)

def _vectorized_sa(a, b, p_init, time_limit, start_time, num_chains=50, batch_size=100):
    """
    Run parallel simulated annealing with vectorized delta computation.
    """
    n = len(p_init)
    
    # Initialize population
    # p: (num_chains, n) each row is a permutation
    p = np.empty((num_chains, n), dtype=np.int64)
    for c in range(num_chains):
        if c == 0:
            p[c] = p_init
        else:
            # Add some noise to other chains
            p[c] = p_init.copy()
            for _ in range(5):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[c, i], p[c, j] = p[c, j], p[c, i]
    
    # Compute initial costs
    costs = np.array([_qap_cost(a, b, p[c]) for c in range(num_chains)], dtype=np.int64)
    
    best_p = p[np.argmin(costs)].copy()
    best_cost = int(np.min(costs))
    
    T_start = 500.0
    T_end = 1.0
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(1.0, elapsed / max(time_limit, 1.0))
        T = T_start * (T_end / T_start) ** frac
        
        # Generate random swaps for each chain
        # For each chain, pick batch_size random swaps
        i_arr = np.random.randint(0, n, size=(num_chains, batch_size))
        j_arr = np.random.randint(0, n, size=(num_chains, batch_size))
        
        # Ensure i != j
        same_mask = i_arr == j_arr
        i_arr[same_mask] = np.random.randint(0, n, size=same_mask.sum())
        # Fix any remaining collisions
        still_same = i_arr == j_arr
        if still_same.any():
            i_arr[still_same] = (i_arr[still_same] + 1) % n
        
        # Compute deltas for all swaps
        # We need to compute deltas for each chain separately because p differs
        # But we can batch within each chain
        
        accepted = np.zeros((num_chains, batch_size), dtype=bool)
        
        for c in range(num_chains):
            # Compute deltas for this chain's swaps
            deltas = _compute_batch_deltas(a, b, p[c], i_arr[c], j_arr[c])
            
            # Acceptance: accept if delta < 0 or with probability exp(-delta/T)
            accept_improve = deltas < 0
            accept_worse = (deltas >= 0) & (np.random.rand(batch_size) < np.exp(-deltas / max(T, 1e-10)))
            
            accepted[c] = accept_improve | accept_worse
            
            # Apply accepted swaps in sequence (to maintain permutation validity)
            # We need to apply them one by one to keep p valid
            for k in range(batch_size):
                if accepted[c, k]:
                    i, j = i_arr[c, k], j_arr[c, k]
                    if i != j:
                        p[c, i], p[c, j] = p[c, j], p[c, i]
                        costs[c] += deltas[k]
            
            # Update global best
            if costs[c] < best_cost:
                best_cost = int(costs[c])
                best_p = p[c].copy()
    
    return best_p, best_cost

def _local_search_polish(a, b, p, time_limit, start_time):
    """
    Final local search polish using vectorized batch evaluation.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    no_improve = 0
    max_no_improve = 100
    batch_size = 200
    
    while time.time() - start_time < time_limit:
        if no_improve >= max_no_improve:
            break
        
        # Generate random swaps
        i_arr = np.random.randint(0, n, size=batch_size)
        j_arr = np.random.randint(0, n, size=batch_size)
        same_mask = i_arr == j_arr
        i_arr[same_mask] = np.random.randint(0, n, size=same_mask.sum())
        still_same = i_arr == j_arr
        if still_same.any():
            i_arr[still_same] = (i_arr[still_same] + 1) % n
        
        # Compute deltas
        deltas = _compute_batch_deltas(a, b, p, i_arr, j_arr)
        
        # Find best improving swap
        improving = deltas < 0
        if improving.any():
            best_idx = np.argmin(np.where(improving, deltas, np.inf))
            i, j = i_arr[best_idx], j_arr[best_idx]
            if i != j:
                p[i], p[j] = p[j], p[i]
                current_cost += deltas[best_idx]
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = int(current_cost)
                    best_p = p.copy()
        else:
            no_improve += 1
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 145.0  # Safety margin before 150s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    # Phase 1: Hungarian initialization
    p_init = _hungarian_initial(a, b)
    
    # Phase 2: Vectorized parallel SA
    elapsed = time.time() - start_time
    sa_time = min(100.0, total_deadline - elapsed - 20)
    if sa_time > 5.0:
        best_p, best_cost = _vectorized_sa(a, b, p_init, sa_time, start_time, num_chains=30, batch_size=50)
    else:
        best_p = p_init
        best_cost = _qap_cost(a, b, p_init)
    
    # Phase 3: Local search polish
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 3:
        ls_time = total_deadline - elapsed - 2
        p_final, cost_final = _local_search_polish(a, b, best_p.copy(), ls_time, start_time)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
    
    return best_p.tolist()
