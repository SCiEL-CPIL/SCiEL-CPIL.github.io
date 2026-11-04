"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_cost_delta(a, b, p, i, j):
    """Compute the change in cost when swapping facilities at positions i and j in p."""
    # p is the current permutation: facility at position k is p[k]
    # We want to swap p[i] and p[j]
    pi = p[i]
    pj = p[j]
    
    # The cost contribution involving facility i (currently at position i) and facility j (currently at position j)
    # When we swap, facility pi moves to position j and facility pj moves to position i
    
    # Delta = new_cost - old_cost
    # We need to compute the change in sum_{k,l} a[k,l] * b[p[k], p[l]]
    
    # Terms that change: those where k or l is i or j
    # Let's compute the contribution of positions i and j before and after
    
    # Before: 
    #   a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj]
    #   + sum_{k!=i,j} a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi]
    #   + sum_{k!=i,j} a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
    
    # After:
    #   a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
    #   + sum_{k!=i,j} a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj]
    #   + sum_{k!=i,j} a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
    
    # Delta = [a[i,i]*(b[pj,pj]-b[pi,pi]) + a[i,j]*(b[pj,pi]-b[pi,pj]) + 
    #          a[j,i]*(b[pi,pj]-b[pj,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])]
    #       + sum_{k!=i,j} [a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + a[k,i]*(b[p[k],pj]-b[p[k],pi]) +
    #                        a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + a[k,j]*(b[p[k],pi]-b[p[k],pj])]
    
    # Let's compute this efficiently using vectorization
    # For the sum over k != i, j:
    #   term_i_k = a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
    #   term_k_i = a[k,i]*(b[p[k],pj] - b[p[k],pi])
    #   term_j_k = a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    #   term_k_j = a[k,j]*(b[p[k],pi] - b[p[k],pj])
    
    # Note: term_i_k + term_k_j = a[i,k]*b[pj,p[k]] - a[i,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi] - a[k,j]*b[p[k],pj]
    # This doesn't simplify nicely, so let's just compute directly.
    
    # Vectorized computation:
    # Create a mask for k != i, j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    # b[pi, p[mask]] and b[pj, p[mask]]
    b_pi = b[pi, p[mask]]
    b_pj = b[pj, p[mask]]
    b_p_mask_pi = b[p[mask], pi]
    b_p_mask_pj = b[p[mask], pj]
    
    a_i_mask = a[i, mask]
    a_mask_i = a[mask, i]
    a_j_mask = a[j, mask]
    a_mask_j = a[mask, j]
    
    delta_sum = np.sum(
        a_i_mask * (b_pj - b_pi) + 
        a_mask_i * (b_p_mask_pj - b_p_mask_pi) +
        a_j_mask * (b_pi - b_pj) +
        a_mask_j * (b_p_mask_pi - b_p_mask_pj)
    )
    
    # Diagonal and cross terms for i, j:
    delta_diag = (
        a[i, i] * (b[pj, pj] - b[pi, pi]) +
        a[i, j] * (b[pj, pi] - b[pi, pj]) +
        a[j, i] * (b[pi, pj] - b[pj, pi]) +
        a[j, j] * (b[pi, pi] - b[pj, pj])
    )
    
    return int(delta_sum + delta_diag)


def _sa_optimize(a, b, p_init, time_limit=10.0, seed=None):
    """Simulated annealing with time limit."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    n = len(p_init)
    p = list(p_init)
    current_cost = _qap_cost(a, b, p)
    best_p = list(p)
    best_cost = current_cost
    
    start_time = time.time()
    # Start with a relatively high temperature and cool down
    # Use a cooling schedule based on time
    T = 10000.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick a random pair to swap
        i, j = rng.sample(range(n), 2)
        
        delta = _swap_cost_delta(a, b, p, i, j)
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept the move
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = list(p)
        
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
    
    return best_p, best_cost


def _hill_climb(a, b, p, time_limit=5.0):
    """2-opt hill climbing: repeatedly swap pairs that improve cost until no improvement."""
    n = len(p)
    p = list(p)
    current_cost = _qap_cost(a, b, p)
    start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            for j in range(i + 1, n):
                delta = _swap_cost_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    break
            if improved:
                continue
        # If no improvement in this full pass, we're at a local optimum
        if not improved:
            break
    
    return p, current_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    total_time_limit = 115.0  # Leave 5s margin for overhead
    
    # Initial solution: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    p_init_list = p_init.tolist()
    
    best_p = p_init_list
    best_cost = _qap_cost(a, b, p_init)
    
    # Multiple SA restarts with different initializations
    n_restarts = 6
    restart_time = (total_time_limit - 20.0) / n_restarts  # Reserve 20s for final hill climb
    
    for r in range(n_restarts):
        remaining = total_time_limit - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        if r == 0:
            p_current = p_init_list
        else:
            # Perturb the best solution found so far
            p_current = list(best_p)
            # Apply random swaps
            n_swaps = max(1, n // 10)
            rng = random.Random(r * 1000 + 42)
            for _ in range(n_swaps):
                i, j = rng.sample(range(n), 2)
                p_current[i], p_current[j] = p_current[j], p_current[i]
        
        sa_time = min(restart_time, remaining - 5.0)
        if sa_time < 1.0:
            break
        
        p_sa, cost_sa = _sa_optimize(a, b, p_current, time_limit=sa_time, seed=r * 7919 + 1)
        
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa
    
    # Final hill climbing polish
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 2.0:
        p_final, cost_final = _hill_climb(a, b, best_p, time_limit=remaining - 1.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p
# EVOLVE-BLOCK-END
