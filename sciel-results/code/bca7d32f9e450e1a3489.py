"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP with O(n^2) per scan.
    Precomputes C_3, C_4 once per scan and updates after swaps.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute diagonal of a (constant)
    a_diag = np.diag(a).copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Precompute C_3 and C_4 once per scan
        C_3 = np.sum(a * D, axis=1)
        C_4 = np.sum(a.T * D, axis=1)
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            V_1 = a[i, :] @ D
            V_2 = a.T[i, :] @ D
            C_1 = a @ D[:, i]
            C_2 = a.T @ D[:, i]
            
            V_sum = V_1 + C_1 + V_2 + C_2 - C_3 - C_4
            
            js = np.arange(i + 1, n)
            if len(js) == 0:
                continue
            
            a_ii = a[i, i]
            D_ii = D[i, i]
            
            a_js_i = a[js, i]
            D_js_i = D[js, i]
            
            T_i_vec = 2 * a_ii * D_js_i + 2 * a_js_i * D_ii - 2 * a_js_i * D_js_i
            
            a_i_js = a[i, js]
            a_js_js = a_diag[js]
            D_js_js = np.diag(D[np.ix_(js, js)])
            D_i_js = D[i, js]
            
            T_j_vec = (a_i_js + a_js_i) * D_js_js + 2 * a_js_js * (D_i_js - D_js_js)
            
            Delta_direct = (a_ii - a_js_js) * (D_js_js - D_ii) + (a_i_js - a_js_i) * (D_js_i - D_i_js)
            
            V_sum_js = V_sum[js]
            
            deltas = V_sum_js - T_i_vec - T_j_vec + Delta_direct
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
                
                # Recompute C_3, C_4 after swap (only when improvement found)
                C_3 = np.sum(a * D, axis=1)
                C_4 = np.sum(a.T * D, axis=1)
        
        if not improved:
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

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer for safety
    
    best_p = None
    best_cost = np.inf
    
    # Initial phase: Random restarts with local search
    num_initials = 4
    time_per_initial = (time_budget * 0.25) / num_initials
    
    for _ in range(num_initials):
        if time.time() - start_time > time_budget * 0.25:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_per_initial, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # ILS phase
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5 or best_p is None:
        return best_p.tolist() if best_p is not None else list(range(n))
        
    consecutive_no_improve = 0
    base_perturb = 2
    
    while time.time() - start_time < time_budget - 3:
        if time.time() - start_time > time_budget - 3:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 8:
            perturb_k = min(12, base_perturb + consecutive_no_improve // 3)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        duration = (time_budget - 3) - (time.time() - start_time)
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
