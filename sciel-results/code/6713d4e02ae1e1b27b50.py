"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    n = len(p)
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _vectorized_2opt_deltas(a, b, p, i):
    """
    For position i in permutation p, compute the delta cost for swapping p[i] with p[j] for all j > i.
    Returns an array of deltas of length n-1-i.
    """
    n = len(p)
    D = b[np.ix_(p, p)]
    A_sym = a + a.T
    
    js = np.arange(i + 1, n)
    
    # Delta for swapping i and j:
    # We need to compute change in cost when we swap positions i and j.
    # The cost is sum_{k,l} a[k,l] * b[p[k],p[l]]
    # After swapping, p'[i]=p[j], p'[j]=p[i], others same.
    # The change involves terms where k or l is i or j.
    #
    # Let's use the standard formula. For a swap of positions i and j:
    # delta = sum_{k != i, j} (a[i,k] + a[j,k] - a[i,i] - a[j,j]) * (b[p[i],p[k]] - b[p[j],p[k]]) 
    #       ... this gets complicated. Let's use a cleaner derivation.
    #
    # Actually, a well-known efficient delta for 2-opt in QAP:
    # Let's define the contribution of pair (i,j) to the cost.
    # A simpler approach: compute the delta by considering the edges.
    #
    # Let's use the formula:
    # delta(i,j) = sum_{k != i,j} (A_sym[i,k] * (D[j,k] - D[i,k]) + A_sym[j,k] * (D[i,k] - D[j,k]))
    #             + (a[i,j] - a[j,i]) * (D[i,j] - D[j,i])
    #             + ... diagonal terms
    #
    # Actually, let's just recompute the affected parts.
    # The total cost is sum_{i,j} a[i,j] * b[p[i],p[j]].
    # When we swap positions i and j, the new permutation p' has p'[i]=p[j], p'[j]=p[i].
    # The cost change is:
    # delta = sum_{k} (a[i,k] * (b[p[j],p[k]] - b[p[i],p[k]]) + a[j,k] * (b[p[i],p[k]] - b[p[j],p[k]]))
    #       + sum_{k} (a[k,i] * (b[p[k],p[j]] - b[p[k],p[i]]) + a[k,j] * (b[p[k],p[i]] - b[p[k],p[j]]))
    #       - 2 * (a[i,j] * b[p[i],p[j]] + a[j,i] * b[p[j],p[i]])  # remove old cross terms
    #       + 2 * (a[i,j] * b[p[j],p[i]] + a[j,i] * b[p[i],p[j]])  # add new cross terms... wait
    #
    # This is getting messy. Let's use the vectorized approach from the current program which seems to work.
    # The current program uses:
    # S_partial = (D[js] - D[i]) @ A_sym[i]
    # S_corrected = S_partial - term_k_i - term_k_j
    # diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
    # cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
    # deltas = S_corrected + diag_term + cross_term
    #
    # Let's verify this is correct. If it's working (score 2.1977), let's keep it but optimize.
    
    S_partial = (D[js] - D[i]) @ A_sym[i]
    
    term_k_i = A_sym[i, i] * (D[js, i] - D[i, i])
    term_k_j = A_sym[i, js] * (D[js, js] - D[i, js])
    
    S_corrected = S_partial - term_k_i - term_k_j
    
    diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
    cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
    
    deltas = S_corrected + diag_term + cross_term
    
    return deltas

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    A_sym = a + a.T
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            deltas = _vectorized_2opt_deltas(a, b, p, i)
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < -1e-9:
                js = np.arange(i + 1, n)
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
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def _greedy_init(a, b):
    """
    Greedy construction heuristic.
    """
    n = len(a)
    p = np.zeros(n, dtype=int)
    used = np.zeros(n, dtype=bool)
    
    # Start with facility 0 at location 0
    p[0] = 0
    used[0] = True
    
    for pos in range(1, n):
        # Find the best facility to place at position pos
        best_fac = -1
        best_cost = np.inf
        
        for fac in range(n):
            if used[fac]:
                continue
            
            # Compute incremental cost of placing fac at pos
            # This involves interactions with all previously placed facilities
            # For simplicity, use a greedy approximation:
            # cost = sum_{j < pos} (a[fac, p[j]] + a[p[j], fac]) * b[pos, p[j]]
            
            prev_positions = np.where(~used)[0]
            prev_positions = prev_positions[prev_positions < pos]
            
            if len(prev_positions) == 0:
                inc_cost = 0
            else:
                inc_cost = np.sum((a[fac, p[prev_positions]] + a[p[prev_positions], fac]) * b[pos, p[prev_positions]])
            
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_fac = fac
        
        p[pos] = best_fac
        used[best_fac] = True
    
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Greedy initialization
    p_greedy = _greedy_init(a, b)
    p_ls, cost_ls = _local_search(a, b, p_greedy, time_budget * 0.15, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # Phase 2: Multiple random restarts
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    num_restarts = max(3, int(remaining / 10))
    time_per_restart = remaining / num_restarts
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_budget * 0.6:
            break
        
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_per_restart, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 3: ILS with adaptive perturbation
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    
    if remaining < 5:
        return best_p.tolist()
    
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 2:
        if time.time() - start_time > time_budget - 2:
            break
        
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
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
