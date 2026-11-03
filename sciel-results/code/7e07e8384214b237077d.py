"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping facilities at positions i and j.
    Optimized O(N) calculation.
    """
    u = p[i]
    v = p[j]
    
    # If i == j, no change
    if i == j:
        return 0
        
    # Direct terms (i,i), (j,j), (i,j), (j,i)
    # Old: a[i,i]*b[u,u] + a[j,j]*b[v,v] + a[i,j]*b[u,v] + a[j,i]*b[v,u]
    # New: a[i,i]*b[v,v] + a[j,j]*b[u,u] + a[i,j]*b[v,u] + a[j,i]*b[u,v]
    direct_delta = (a[i, i] * (b[v, v] - b[u, u]) + 
                    a[j, j] * (b[u, u] - b[v, v]) + 
                    a[i, j] * (b[v, u] - b[u, v]) + 
                    a[j, i] * (b[u, v] - b[v, u]))
    
    # Cross terms
    # Sum over k != i, j of:
    # (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) +
    # (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
    
    # Vectorized calculation for all k
    # We need to exclude i and j from the sum
    # p is the permutation array. p[k] is the location of facility k.
    # We want sum over k where k != i and k != j.
    
    # Create vectors
    # diff_a_row = a[i, :] - a[j, :]  (shape N)
    # diff_b_col = b[v, :] - b[u, :]  (shape N)
    # term1 = diff_a_row[p] * diff_b_col[p] ? No.
    # b[v, p[k]] is element p[k] of row v of b.
    # So we need b[v, p] and b[u, p].
    
    b_v_p = b[v, p]
    b_u_p = b[u, p]
    diff_b = b_v_p - b_u_p
    
    a_i_row = a[i, :]
    a_j_row = a[j, :]
    diff_a_row = a_i_row - a_j_row
    
    # term1_k = (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    term1 = diff_a_row * diff_b
    
    # Similarly for the other part:
    # (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
    # a[:, i] - a[:, j]
    a_col_i = a[:, i]
    a_col_j = a[:, j]
    diff_a_col = a_col_i - a_col_j
    
    b_p_v = b[p, v]
    b_p_u = b[p, u]
    diff_b_col = b_p_v - b_p_u
    
    term2 = diff_a_col * diff_b_col
    
    total_cross = np.sum(term1 + term2)
    
    # Subtract the terms for k=i and k=j which were included in the sum
    # For k=i:
    # term1[i] = (a[i,i] - a[j,i]) * (b[v, p[i]] - b[u, p[i]]) = (a[i,i] - a[j,i]) * (b[v, u] - b[u, u])
    # term2[i] = (a[i,i] - a[i,j]) * (b[p[i], v] - b[p[i], u]) = (a[i,i] - a[i,j]) * (b[u, v] - b[u, u])
    
    # For k=j:
    # term1[j] = (a[i,j] - a[j,j]) * (b[v, p[j]] - b[u, p[j]]) = (a[i,j] - a[j,j]) * (b[v, v] - b[u, v])
    # term2[j] = (a[j,i] - a[j,j]) * (b[p[j], v] - b[p[j], u]) = (a[j,i] - a[j,j]) * (b[v, v] - b[v, u])
    
    # It is safer to just compute the sum for all k and subtract i and j contributions
    # Or simply mask them out. Masking is O(N) but avoids algebra errors.
    # Given N=256, masking is very fast.
    
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    cross_sum = np.sum((term1 + term2)[mask])
    
    return int(direct_delta + cross_sum)

def _two_opt_search(a, b, p, time_limit, start_time):
    """
    Perform 2-opt local search with first-improvement strategy.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break  # First improvement: restart scan from beginning
        
        if not improved:
            break
    
    return best_p, best_cost

def _greedy_init(a, b):
    """
    Greedy initialization: assign facilities to locations one by one.
    """
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    
    # Order facilities by row sum (high flow first)
    a_row_sums = a.sum(axis=1)
    order = np.argsort(-a_row_sums)
    
    for idx in order:
        best_loc = -1
        best_cost = float('inf')
        
        for loc in range(n):
            if assigned[loc]:
                continue
            
            # Incremental cost
            # sum_{j assigned} a[idx, j] * b[loc, p[j]] + a[j, idx] * b[p[j], loc]
            # Vectorized:
            if np.any(assigned):
                assigned_indices = np.where(assigned)[0]
                # p[assigned_indices] are the locations
                inc_cost = np.sum(a[idx, assigned_indices] * b[loc, p[assigned_indices]]) + \
                           np.sum(a[assigned_indices, idx] * b[p[assigned_indices], loc])
            else:
                inc_cost = 0
            
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_loc = loc
        
        p[idx] = best_loc
        assigned[best_loc] = True
    
    return p

def _perturb(p, n_swaps, rng):
    """
    Perform random swaps on the permutation.
    """
    p = p.copy()
    n = len(p)
    for _ in range(n_swaps):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    """
    Solve the QAP instance tai256c using Iterated Local Search (ILS).
    """
    start_time = time.time()
    n = len(a)
    
    # Time budget: 150s limit, use 145s for safety
    time_budget = 145
    
    # 1. Initialization
    p_init = _greedy_init(a, b)
    best_p, best_cost = _two_opt_search(a, b, p_init, time_limit=10, start_time=start_time)
    
    rng = np.random.default_rng(42)
    
    # 2. Iterated Local Search
    # We will iterate: Perturb -> Local Search -> Accept
    # We increase perturbation strength if stuck
    
    current_p = best_p.copy()
    current_cost = best_cost
    stagnant_count = 0
    base_swaps = 5
    
    while time.time() - start_time < time_budget - 10:
        # Determine number of swaps
        # Increase swaps if stuck
        if stagnant_count > 10:
            n_swaps = base_swaps + stagnant_count // 5
        else:
            n_swaps = base_swaps
        
        # Perturb
        p_perturbed = _perturb(current_p, n_swaps, rng)
        
        # Local Search
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed - 5
        if remaining < 5:
            break
            
        # Limit local search time per iteration to ensure we do many iterations
        # But also allow it to converge.
        ls_time_limit = min(remaining, 10) 
        
        p_new, cost_new = _two_opt_search(a, b, p_perturbed, time_limit=ls_time_limit, start_time=start_time)
        
        # Acceptance
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            current_p = p_new.copy()
            current_cost = cost_new
            stagnant_count = 0
        else:
            stagnant_count += 1
            # Optionally, revert to best if perturbation was too destructive?
            # ILS usually keeps the current solution for further perturbation if it's not worse than the *previous* current, 
            # but here we compare to global best. 
            # If we don't update current_p, we perturb the same best_p again.
            # This is "First Fit" or "Best Fit" ILS.
            # Let's stick to perturbing the global best to ensure we explore around the best found.
            pass
    
    # Final polish
    if time.time() - start_time < time_budget - 2:
        remaining = time_budget - 2 - (time.time() - start_time)
        if remaining > 1:
            best_p, best_cost = _two_opt_search(a, b, best_p, time_limit=remaining, start_time=start_time)

    if best_p is None:
        best_p = np.arange(n)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
