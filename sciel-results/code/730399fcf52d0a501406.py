"""QAP tai256c solver: ILS with Vectorized 2-Opt and Hungarian-Driven Diversification."""
import numpy as np
import time
import random
import scipy.optimize as opt

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP with best-improvement per pass.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    def _compute_deltas(i, js, D, A_sym, A_diag, a):
        """Compute swap deltas for position i against all positions in js."""
        # Vectorized calculation of delta for swapping i with all j in js
        # Delta = sum_{k != i, j} a[i,k]b[p[i],p[k]] + a[k,i]b[p[k],p[i]] + ... - old terms
        # Simplified formula for symmetric A:
        # diff_A = A_sym[i] - A_sym[js]
        # diff_D = D[js] - D[i]
        # S = sum(diff_A * diff_D)
        # Corrections for diagonal and cross terms
        
        diff_A = A_sym[i] - A_sym[js]
        diff_D = D[js] - D[i]
        S = np.sum(diff_A * diff_D, axis=1)
        
        # Term involving i and i (diagonal in A, but off-diagonal in D swap logic)
        # The standard vectorized delta for QAP 2-opt:
        # Let u = i, v = j
        # delta = sum_k (a[u,k] - a[v,k]) * (b[p[u], p[k]] - b[p[v], p[k]]) + ...
        
        # Using the specific correction terms derived in the current program:
        coeff_i = A_sym[i, i] - A_sym[js, i]
        term_i = coeff_i * (D[js, i] - D[i, i])
        
        coeff_j = A_sym[i, js] - A_diag[js]
        term_j = coeff_j * (D[js, js] - D[i, js])
        
        diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
        cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
        
        return S - term_i - term_j + diag_term + cross_term
    
    def _perform_swap(i, j, p, D):
        p[i], p[j] = p[j], p[i]
        D[i], D[j] = D[j].copy(), D[i].copy()
        D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        # Best-improvement pass: find the single best swap in the entire permutation
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            deltas = _compute_deltas(i, js, D, A_sym, A_diag, a)
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < best_delta:
                best_delta = min_delta
                best_i = i
                best_j = js[min_idx]
        
        if best_i >= 0 and best_delta < 0:
            _perform_swap(best_i, best_j, p, D)
            current_cost += int(best_delta)
            improved = True
        else:
            break
    
    return p, current_cost

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    """
    n = len(a)
    # Surrogate cost: Linearization of the quadratic objective
    # cost[i, j] approx sum_k a[i, k] * b[j, k]
    cost_matrix = a @ b
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _rank_matching_init(a, b):
    """
    Initialize by matching high-flow facilities to close locations.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    # Sort facilities by flow intensity (descending)
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by centrality (ascending distance sum, i.e., low sum is 'central' if b is distance)
    # Note: In QAP, b is usually distance, so smaller sum is better.
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _hungarian_reopt(a, b, p):
    """
    Perform a global re-optimization using the Hungarian algorithm.
    Constructs a linearized cost matrix based on the current assignment p.
    This allows large-scale structural changes (global adjustments) as opposed to local swaps.
    """
    n = len(p)
    # Linearization:
    # Cost of assigning facility i to location j, given current assignment p:
    # C[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
    
    # Efficient calculation:
    # Term 1: (a @ b[:, p])[i, j] ? No.
    # Let B_p be b with columns permuted by p? No, b[p[k], :] is row p[k].
    # We want M1[i, j] = sum_k a[i, k] * b[j, p[k]]
    # This is (a @ B_perm)[i, j] where B_perm[k, j] = b[j, p[k]]?
    # Let's construct B_perm such that B_perm[k, :] = b[:, p[k]]? No.
    # b[j, p[k]] is the element at row j, col p[k].
    # Let P be permutation matrix. B_perm = b @ P? No.
    # b[j, p[k]] -> If we take b.T, it's b.T[p[k], j].
    # So M1 = a @ (b.T[:, p]).T ?
    # Let's just compute it directly using matrix multiplication for speed.
    # M1[i, j] = sum_k a[i, k] * b[j, p[k]]
    # Let B_col = b[:, p] (columns of b permuted by p)
    # Then B_col.T[j, k] = b[p[k], j] ? No, B_col.T[k, j] = b[j, p[k]].
    # So M1 = a @ B_col.T
    
    B_col = b[:, p]  # Shape (n, n), column k is b[:, p[k]]
    M1 = a @ B_col.T # Shape (n, n), M1[i, j] = sum_k a[i, k] * B_col.T[k, j] = sum_k a[i, k] * b[j, p[k]]
    
    # M2[i, j] = sum_k a[k, i] * b[p[k], j]
    # Let B_row = b[p, :] (rows of b permuted by p)
    # B_row[k, j] = b[p[k], j]
    # M2 = a.T @ B_row
    B_row = b[p, :]
    M2 = a.T @ B_row
    
    C = M1 + M2
    
    row_ind, col_ind = opt.linear_sum_assignment(C)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    return new_p

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def _reversal_perturb(p, k):
    """Apply k random reversals (segment reversals) for stronger diversification."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = sorted(random.sample(range(n), 2))
        p[i:j+1] = p[i:j+1][::-1]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115.0  # Leave 5s buffer for safety under 120s limit
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Diversified Initial Solutions
    # 1.1 Hungarian Initialization
    p_hung = _hungarian_init(a, b)
    ls_limit = time_budget * 0.1
    p_ls, cost_ls = _local_search(a, b, p_hung, ls_limit, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 1.2 Rank Matching Initialization
    p_rank = _rank_matching_init(a, b)
    remaining_for_rank = time_budget * 0.2 - (time.time() - start_time)
    if remaining_for_rank > 2:
        p_ls_r, cost_ls_r = _local_search(a, b, p_rank, remaining_for_rank, start_time)
        if cost_ls_r < best_cost:
            best_cost = cost_ls_r
            best_p = p_ls_r.copy()

    # 1.3 Random Initialization
    p_rand = np.random.permutation(n)
    remaining_for_rand = time_budget * 0.3 - (time.time() - start_time)
    if remaining_for_rand > 2:
        p_ls_rand, cost_ls_rand = _local_search(a, b, p_rand, remaining_for_rand, start_time)
        if cost_ls_rand < best_cost:
            best_cost = cost_ls_rand
            best_p = p_ls_rand.copy()

    # Phase 2: ILS with Adaptive Diversification
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 10:
        # Final polish
        p_final, cost_final = _local_search(a, b, best_p, time_budget - 2, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
        return best_p.tolist()
    
    consecutive_no_improve = 0
    iteration = 0
    
    while time.time() - start_time < time_budget - 5:
        iteration += 1
        
        # Determine diversification strategy
        # If stuck, use stronger perturbations or Hungarian re-optimization
        use_hungarian = False
        perturb_k = 4
        
        if consecutive_no_improve > 8:
            use_hungarian = True
            perturb_k = 10
        elif consecutive_no_improve > 4:
            perturb_k = 8
        else:
            perturb_k = 4
            
        if use_hungarian:
            # Structural Change: Global adjustment via Hungarian Re-optimization
            # This solves a linear assignment problem to propose a new structure
            p_new = _hungarian_reopt(a, b, best_p)
            
            # Add some random noise to avoid cycling
            if random.random() < 0.5:
                p_new = _perturb(p_new, perturb_k)
        else:
            # Standard local perturbation
            if consecutive_no_improve % 3 == 0 and consecutive_no_improve > 0:
                p_new = _reversal_perturb(best_p, perturb_k)
            else:
                p_new = _perturb(best_p, perturb_k)
        
        # Allocate time for local search on the new candidate
        # Reserve 5s for final polish and overhead
        time_for_ls = (time_budget - 5) - (time.time() - start_time)
        if time_for_ls < 1:
            break
            
        p_ls, cost_ls = _local_search(a, b, p_new, time_for_ls, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    # Phase 3: Final Polish
    elapsed = time.time() - start_time
    if elapsed < time_budget - 2:
        p_final, cost_final = _local_search(a, b, best_p, time_budget - 1, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p.tolist()
