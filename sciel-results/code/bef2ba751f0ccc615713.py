"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        diff_A = A_sym[i] - A_sym[js]
        diff_D = D[js] - D[i]
        S = np.sum(diff_A * diff_D, axis=1)
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

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    Uses the row-wise sum of A*B as a linearization surrogate.
    """
    n = len(a)
    # Surrogate: for each facility i, the cost of assigning it to location j
    # is approximately sum_k a[i,k] * b[j, p[k]] but we don't know p yet.
    # Use a linearization: cost[i,j] = sum_k a[i,k] * b[j,k] (treating B as identity mapping)
    # Better: use the row sums of A weighted by column sums of B
    cost_matrix = a @ b
    
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    
    return p

def _diversified_random_init(n, seed=None):
    """Generate a diversified random initialization."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(n)

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer for safety
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Diversified initial solutions
    # 1. Hungarian Initialization (structural)
    p_hung = _hungarian_init(a, b)
    ls_limit_hung = time_budget * 0.12
    p_ls, cost_ls = _local_search(a, b, p_hung, ls_limit_hung, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 2. Multiple random restarts for diversity
    elapsed = time.time() - start_time
    remaining_for_randoms = time_budget * 0.25 - elapsed
    if remaining_for_randoms > 3:
        num_randoms = 3
        time_per_random = remaining_for_randoms / num_randoms
        for r in range(num_randoms):
            if time.time() - start_time > time_budget * 0.25:
                break
            p_init = _diversified_random_init(n, seed=42 + r * 1000)
            p_ls_r, cost_ls_r = _local_search(a, b, p_init, time_per_random, start_time)
            if cost_ls_r < best_cost:
                best_cost = cost_ls_r
                best_p = p_ls_r.copy()
    
    # Phase 2: ILS with adaptive perturbation
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 10:
        # Not enough time for ILS, do a final polish
        p_final, cost_final = _local_search(a, b, best_p, time_budget - 2, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
        return best_p.tolist()
    
    consecutive_no_improve = 0
    base_perturb = 4
    
    while time.time() - start_time < time_budget - 5:
        # Check time with safety margin
        if time.time() - start_time > time_budget - 5:
            break
            
        # Adaptive perturbation: increase when stuck
        if consecutive_no_improve > 12:
            perturb_k = min(25, base_perturb + consecutive_no_improve)
        elif consecutive_no_improve > 6:
            perturb_k = min(18, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
        
        # Alternate between swap and reversal perturbations for diversity
        if consecutive_no_improve % 3 == 0 and consecutive_no_improve > 0:
            p_pert = _reversal_perturb(best_p, max(1, perturb_k // 2))
        else:
            p_pert = _perturb(best_p, perturb_k)
        
        # Allocate time for this LS run
        duration = (time_budget - 5) - (time.time() - start_time)
        if duration < 1:
            break
            
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    # Phase 3: Final polish on the best solution
    elapsed = time.time() - start_time
    if elapsed < time_budget - 2:
        p_final, cost_final = _local_search(a, b, best_p, time_budget - 1, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
