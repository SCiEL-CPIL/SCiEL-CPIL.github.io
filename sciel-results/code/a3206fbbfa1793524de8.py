"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP.
    Uses vectorized delta calculation.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    A_sym = a + a.T
    
    # Precompute diagonal of A_sym for correction terms? 
    # We need A_sym[i, js] and np.diag(A_sym[js])
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Corrections
            # term_i: k=i part
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # term_j: k=j part
            # A_sym[i, js] is (m,)
            # np.diag(A_sym[js]) is (m,)
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
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = np.random.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def _greedy_init(a, b, n):
    """
    Simple greedy initialization:
    Assign facilities one by one to the location that minimizes the current cost increase.
    This is O(n^3) which is acceptable for n=256.
    """
    p = np.zeros(n, dtype=int)
    assigned = np.zeros(n, dtype=bool)
    
    # To speed up, we can just do random if this is too slow, but let's try.
    # Actually, full greedy is complex to vectorize efficiently without loops.
    # Let's use a simpler heuristic: Random.
    # The directive says row sums are weak, so random is a safe baseline.
    return np.random.permutation(n)

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multiple random restarts with local search
    # We allocate 30% of time to this phase
    phase1_time = time_budget * 0.3
    num_initials = 10
    time_per_initial = phase1_time / num_initials
    
    for _ in range(num_initials):
        if time.time() - start_time > start_time + phase1_time:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_per_initial, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: ILS
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5 or best_p is None:
        return best_p.tolist() if best_p is not None else np.random.permutation(n).tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 2:
        if time.time() - start_time > time_budget - 2:
            break
            
        # Adaptive perturbation: increase if stuck
        if consecutive_no_improve > 5:
            perturb_k = min(20, base_perturb + consecutive_no_improve)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Give local search the remaining time minus a small buffer
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
