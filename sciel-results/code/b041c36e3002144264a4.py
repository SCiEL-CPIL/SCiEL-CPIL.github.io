"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute exact QAP cost for assignment p."""
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _hungarian_init(a, b, n):
    """
    Initialize using Linear Assignment Problem (LAP) on row/col sums.
    This is a standard and effective heuristic for QAP, matching high flow
    facilities to low distance locations.
    """
    from scipy.optimize import linear_sum_assignment
    
    # Calculate row sums (proxy for flow and distance weights)
    r_a = np.sum(a, axis=1)
    r_b = np.sum(b, axis=1)
    
    # Cost matrix for LAP: C[i, j] = r_a[i] * r_b[j]
    # We want to minimize sum C[i, p[i]]
    cost_matrix = np.outer(r_a, r_b)
    
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=int)
        p[row_ind] = col_ind
        return p
    except Exception:
        return np.random.permutation(n)

def _local_search(a, b, p, time_limit, start_time):
    """
    Efficient 2-opt local search with precomputed cost deltas.
    Uses the formula: delta(i,j) = sum_{k != i,j} (a[i,k] + a[k,i]) * (D[j,k] - D[i,k])
    + (a[i,i] - a[j,j])*(D[j,j] - D[i,i]) + (a[i,j] - a[j,i])*(D[j,i] - D[i,j])
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute A_sym = A + A.T for efficient delta calculation
    A_sym = a + a.T
    
    # Precompute row sums of A_sym for faster delta computation
    # delta(i,j) ≈ sum_k A_sym[i,k] * D[j,k] - sum_k A_sym[j,k] * D[i,k] - correction terms
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation
            # S = sum_k (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
            diff_A = A_sym[i] - A_sym[js]  # (n-i-1, n)
            diff_D = D[js] - D[i]          # (n-i-1, n)
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction for k=i and k=j terms
            # k=i term: (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # k=j term: (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
            coeff_j = A_sym[i, js] - np.diag(A_sym[js])
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # Direct terms
            diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D rows and columns
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _perturb_3opt(p, k):
    """Apply k random 3-opt moves for better diversification."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        if n < 3:
            break
        i, j, k_idx = random.sample(range(n), 3)
        # Reverse the segment between i and k_idx (assuming i < k_idx)
        if i > k_idx:
            i, k_idx = k_idx, i
        p[i:k_idx+1] = p[i:k_idx+1][::-1]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer for overhead
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Generate diverse initial solutions
    # Use eigenvector-based init + random restarts
    num_initials = 8
    phase1_time = time_budget * 0.25
    
    # First: try Hungarian-based initialization
    p_hung = _hungarian_init(a, b, n)
    p_ls, cost_ls = _local_search(a, b, p_hung, phase1_time / 2, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # Then: random restarts
    remaining_phase1 = phase1_time - (time.time() - start_time)
    if remaining_phase1 > 2:
        num_random = min(6, int(remaining_phase1 / 3))
        for _ in range(num_random):
            if time.time() - start_time > start_time + phase1_time:
                break
            p_init = np.random.permutation(n)
            time_per_ls = min(3.0, (phase1_time - (time.time() - start_time)) / max(1, num_random - _))
            p_ls, cost_ls = _local_search(a, b, p_init, time_per_ls, start_time)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
    
    # Phase 2: Iterated Local Search with 3-opt perturbation
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5 or best_p is None:
        return best_p.tolist() if best_p is not None else np.random.permutation(n).tolist()
    
    consecutive_no_improve = 0
    base_perturb = 2
    
    while time.time() - start_time < time_budget - 2:
        if time.time() - start_time > time_budget - 2:
            break
        
        # Adaptive perturbation: increase if stuck
        if consecutive_no_improve > 8:
            perturb_k = min(12, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb_3opt(best_p, perturb_k)
        
        # Allocate time for this LS run
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
            break
            
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
