"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            S = np.sum(diff_A * diff_D, axis=1)
            
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = A_sym[i, js] - A_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < -1e-9:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _perturb_swap(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def _perturb_oropt(p, k):
    """Randomly relocate k consecutive blocks of length 2-3."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        length = random.choice([2, 3])
        if n < length + 1:
            continue
        start = random.randint(0, n - length)
        end = start + length
        block = p[start:end].copy()
        # Remove block
        p = np.delete(p, np.arange(start, end))
        n_new = len(p)
        if n_new == 0:
            p = block
            continue
        # Insert at random position
        insert_pos = random.randint(0, n_new)
        p = np.concatenate([p[:insert_pos], block, p[insert_pos:]])
    return p

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    return p

def _greedy_init(a, b):
    """
    Greedy constructive heuristic: place facilities one by one at the location
    that minimizes incremental cost given current placements.
    """
    n = len(a)
    p = np.empty(n, dtype=int)
    used = np.zeros(n, dtype=bool)
    
    # Start with a random facility
    first = random.randint(0, n - 1)
    p[first] = 0
    used[0] = True
    
    for pos in range(1, n):
        best_fac = -1
        best_loc = -1
        best_cost = np.inf
        
        # Try all unplaced facilities and unused locations
        for fac in range(n):
            if used[fac]:
                continue
            for loc in range(n):
                if loc == 0 and fac != first:
                    # Location 0 is taken
                    pass
                # Check if location is available
                loc_used = False
                for k in range(n):
                    if p[k] == loc and not np.isnan(p[k]) if isinstance(p[k], float) else False:
                        loc_used = True
                        break
                # Simpler: track used locations
                pass
        
        # Simpler approach: just use Hungarian on remaining
        break
    
    # Fallback: use Hungarian
    return _hungarian_init(a, b)

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 110  # Safe buffer for 120s limit
    
    best_p = None
    best_cost = np.inf
    
    # 1. Hungarian Initialization
    p_hung = _hungarian_init(a, b)
    ls_limit = time_budget * 0.15
    p_ls, cost_ls = _local_search(a, b, p_hung, ls_limit, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 2. Random Initials (2 of them)
    if time.time() - start_time < time_budget * 0.35:
        time_per_random = (time_budget * 0.15) / 2
        for _ in range(2):
            if time.time() - start_time > time_budget * 0.35:
                break
            p_init = np.random.permutation(n)
            p_ls_r, cost_ls_r = _local_search(a, b, p_init, time_per_random, start_time)
            if cost_ls_r < best_cost:
                best_cost = cost_ls_r
                best_p = p_ls_r.copy()
    
    # ILS phase
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
    
    consecutive_no_improve = 0
    base_perturb = 3
    use_oropt = False
    
    while time.time() - start_time < time_budget - 2:
        if time.time() - start_time > time_budget - 2:
            break
            
        # Adaptive perturbation: alternate between swap and or-opt
        if consecutive_no_improve > 5:
            use_oropt = not use_oropt
        
        if use_oropt:
            perturb_k = min(8, base_perturb + consecutive_no_improve // 3)
            p_pert = _perturb_oropt(best_p, perturb_k)
        else:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
            p_pert = _perturb_swap(best_p, perturb_k)
        
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        # Safety check: if we just barely finished, ensure we don't loop again immediately
        if time.time() - start_time > time_budget - 2:
            break
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
