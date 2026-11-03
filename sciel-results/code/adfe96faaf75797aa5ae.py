"""
Simulated Annealing solver for QAPLIB wil100.
Combines multi-start greedy+random initialization with steepest-descent local
search and a final SA polish. Uses a vectorized O(n^2) delta computation for
steepest-descent moves to accelerate the local search phase.
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

def _compute_all_deltas_for_i(a, b, p, i):
    """
    Compute swap deltas for a fixed position i against all j > i.
    Returns an array of deltas for j = i+1, ..., n-1.
    """
    n = len(p)
    pi = p[i]
    
    # Precompute for fixed i
    diff_a_col_i = a[:, i]  # a[k, i] for all k
    diff_a_row_i = a[i, :]  # a[i, k] for all k
    
    # We need deltas for all j > i
    # Vectorize over j
    js = np.arange(i + 1, n)
    
    # p[js] gives the locations for all j > i
    p_js = p[js]
    
    # Term 1 for each j: sum_k (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    # = sum_k a[k,i] b[p[k], p[j]] - sum_k a[k,i] b[p[k], p[i]]
    #   - sum_k a[k,j] b[p[k], p[j]] + sum_k a[k,j] b[p[k], p[i]]
    
    # Let Bp = b[p, p] (the permuted distance matrix)
    # sum_k a[k,i] b[p[k], p[j]] = (a[:, i] @ Bp[:, j]) ... no
    # sum_k a[k,i] b[p[k], p[j]] = dot(a[:, i], b[p, p[j]])
    # This is a row of (a[:, i] as row) @ Bp? No.
    # Let's just compute directly for each j in a vectorized way over the array of js.
    
    # For a fixed i and array of js:
    # T1[j] = sum_k (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    
    # This is hard to fully vectorize over j without O(n^2) memory per call.
    # But n=100, so O(n^2) = 10,000 is fine.
    
    # Let's compute T1, T2, T3 for all j > i at once.
    
    # T1[j] for j in js:
    # sum_k (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    
    # Expand:
    # sum_k a[k,i] b[p[k], p[j]] - sum_k a[k,i] b[p[k], p[i]]
    #   - sum_k a[k,j] b[p[k], p[j]] + sum_k a[k,j] b[p[k], p[i]]
    
    # Term A: sum_k a[k,i] b[p[k], p[j]] for each j in js
    # = dot(a[:, i], b[p, p[j]]) for each j
    # b[p, p[j]] is a column of Bp where Bp = b[p, p]
    # Actually b[p, p[j]] is the column of b[p, :] indexed by p[j]
    # Let Bp = b[np.ix_(p, p)]
    # Then b[p[k], p[j]] = Bp[k, j] (since p[j] is the j-th element of p)
    # Wait, Bp[k, m] = b[p[k], p[m]]. So b[p[k], p[j]] = Bp[k, j].
    # So sum_k a[k,i] Bp[k, j] = (a[:, i] @ Bp)[:, j] ... no, a[:, i] is a vector.
    # sum_k a[k,i] Bp[k, j] = dot(a[:, i], Bp[:, j])
    
    # Let's precompute Bp = b[np.ix_(p, p)]
    Bp = b[np.ix_(p, p)]
    
    # A[j] = dot(a[:, i], Bp[:, j]) for j in js
    # This is (a[:, i] @ Bp)[js]
    # a[:, i] @ Bp gives a vector of length n, where element m is sum_k a[k,i] Bp[k,m]
    # We want elements indexed by js.
    A_vec = a[:, i] @ Bp  # shape (n,)
    A_vals = A_vec[js]
    
    # Term B: sum_k a[k,i] b[p[k], p[i]] = dot(a[:, i], Bp[:, i]) = A_vec[i]
    B_val = A_vec[i]
    
    # Term C: sum_k a[k,j] b[p[k], p[j]] for j in js
    # = sum_k a[k,j] Bp[k,j]
    # This is the diagonal of (a @ Bp) at index j?
    # (a @ Bp)[j, m] = sum_k a[j,k] Bp[k,m] = sum_k a[j,k] b[p[k], p[m]]
    # We want sum_k a[k,j] b[p[k], p[j]] = sum_k a[k,j] Bp[k,j]
    # This is (a.T @ Bp)[j, j] = (a @ Bp.T)[j, j]
    # Since Bp is symmetric (b is symmetric), Bp.T = Bp.
    # So C[j] = (a @ Bp)[j, j]
    ABp = a @ Bp  # shape (n, n)
    C_vals = np.diag(ABp)[js]
    
    # Term D: sum_k a[k,j] b[p[k], p[i]] for j in js
    # = sum_k a[k,j] Bp[k, i]
    # = (a.T @ Bp)[:, i][js] = (a @ Bp.T)[:, i][js] = (a @ Bp)[:, i][js]
    D_vals = ABp[js, i]
    
    # So T1[j] = A_vals - B_val - C_vals + D_vals
    T1_full = A_vals - B_val - C_vals + D_vals
    
    # Now subtract the k=i and k=j corrections
    # Correction for k=i: (a[i,i] - a[i,j]) * (b[pi, p[j]] - b[pi, pi])
    # b[pi, p[j]] = Bp[i, j] (since p[i]=pi, so Bp[i, j] = b[p[i], p[j]] = b[pi, p[j]])
    # b[pi, pi] = Bp[i, i]
    # So corr_i[j] = (a[i,i] - a[i,j]) * (Bp[i, j] - Bp[i, i])
    # a[i, j] for j in js: a[i, js]
    # Bp[i, j] for j in js: Bp[i, js]
    corr_i = (a[i, i] - a[i, js]) * (Bp[i, js] - Bp[i, i])
    
    # Correction for k=j: (a[j,i] - a[j,j]) * (b[p[j], p[j]] - b[p[j], p[i]])
    # b[p[j], p[j]] = Bp[j, j]
    # b[p[j], p[i]] = Bp[j, i]
    # So corr_j[j] = (a[j,i] - a[j,j]) * (Bp[j, j] - Bp[j, i])
    # a[j, i] for j in js: a[js, i]
    # a[j, j] for j in js: a[js, js] (diagonal elements)
    # Bp[j, j] for j in js: np.diag(Bp)[js]
    # Bp[j, i] for j in js: Bp[js, i]
    diag_Bp = np.diag(Bp)
    corr_j = (a[js, i] - a[js, js]) * (diag_Bp[js] - Bp[js, i])
    
    T1 = T1_full - corr_i - corr_j
    
    # Term 2 for each j: sum_k (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
    # = sum_k a[i,k] b[p[j], p[k]] - sum_k a[i,k] b[p[i], p[k]]
    #   - sum_k a[j,k] b[p[j], p[k]] + sum_k a[j,k] b[p[i], p[k]]
    
    # b[p[j], p[k]] = Bp[j, k]
    # b[p[i], p[k]] = Bp[i, k]
    
    # Term E: sum_k a[i,k] Bp[j, k] = (a[i, :] @ Bp[j, :]) = (a[i, :] @ Bp)[j] ... no
    # sum_k a[i,k] Bp[j,k] = dot(a[i, :], Bp[j, :])
    # Since Bp is symmetric, Bp[j, k] = Bp[k, j]
    # dot(a[i, :], Bp[j, :]) = (a[i, :] @ Bp)[j]? No.
    # a[i, :] @ Bp gives a vector where element m is sum_k a[i,k] Bp[k,m]
    # We want sum_k a[i,k] Bp[j,k] = sum_k a[i,k] Bp[k,j] (by symmetry) = (a[i,:] @ Bp)[j]
    # So E[j] = (a[i, :] @ Bp)[j]
    a_i_Bp = a[i, :] @ Bp  # shape (n,)
    E_vals = a_i_Bp[js]
    
    # Term F: sum_k a[i,k] b[p[i], p[k]] = sum_k a[i,k] Bp[i,k] = a_i_Bp[i]
    F_val = a_i_Bp[i]
    
    # Term G: sum_k a[j,k] b[p[j], p[k]] = sum_k a[j,k] Bp[j,k]
    # = (a @ Bp)[j, j] (by symmetry, same as before)
    # Wait, (a @ Bp)[j, m] = sum_k a[j,k] Bp[k,m]
    # sum_k a[j,k] Bp[j,k] = sum_k a[j,k] Bp[k,j] (symmetry) = (a @ Bp)[j, j]
    # So G[j] = np.diag(ABp)[j]
    G_vals = np.diag(ABp)[js]
    
    # Term H: sum_k a[j,k] b[p[i], p[k]] = sum_k a[j,k] Bp[i,k]
    # = sum_k a[j,k] Bp[k,i] (symmetry) = (a @ Bp)[j, i]
    H_vals = ABp[js, i]
    
    T2_full = E_vals - F_val - G_vals + H_vals
    
    # Corrections for T2:
    # k=i: (a[i,i] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[i]])
    # = (a[i,i] - a[j,i]) * (Bp[j, i] - Bp[i, i])
    # a[j,i] for j in js: a[js, i]
    # Bp[j, i] for j in js: Bp[js, i]
    corr2_i = (a[i, i] - a[js, i]) * (Bp[js, i] - Bp[i, i])
    
    # k=j: (a[i,j] - a[j,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
    # = (a[i,j] - a[j,j]) * (Bp[j, j] - Bp[i, j])
    # a[i,j] for j in js: a[i, js]
    # a[j,j] for j in js: a[js, js]
    # Bp[j, j]: diag_Bp[js]
    # Bp[i, j] for j in js: Bp[i, js]
    corr2_j = (a[i, js] - a[js, js]) * (diag_Bp[js] - Bp[i, js])
    
    T2 = T2_full - corr2_i - corr2_j
    
    # Term 3 for each j: (a[i,j] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[j]])
    # = (a[i,j] - a[j,i]) * (Bp[j, i] - Bp[i, j])
    T3 = (a[i, js] - a[js, i]) * (Bp[js, i] - Bp[i, js])
    
    deltas = T1 + T2 + T3
    return deltas

def _local_search(a, b, p, max_iter=None, time_limit=None, start_time=None):
    """
    Perform steepest descent local search until no improving move is found
    or limits are reached.
    Returns the improved permutation and cost.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 100000
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    improved = True
    while improved and max_iter > 0:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Check all pairs for the best improving move
        for i in range(n):
            # Compute deltas for all j > i
            deltas = _compute_all_deltas_for_i(a, b, p, i)
            for idx, j in enumerate(range(i + 1, n)):
                delta = deltas[idx]
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
            max_iter -= 1
    
    return p, current_cost

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
    
    # Local search to refine
    p, current_cost = _local_search(a, b, p, max_iter=50000, time_limit=total_deadline - 5, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10:
        # Try a few random starts
        num_random_starts = min(5, int(remaining_time // 10))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 5:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # Local search to refine
            p, current_cost = _local_search(a, b, p, max_iter=50000, time_limit=total_deadline - 5, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Final polish with SA on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        T_start = max(1.0, best_cost * 0.01)
        T_end = 0.1
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
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
