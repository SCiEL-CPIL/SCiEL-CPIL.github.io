"""
Tabu Search solver for QAPLIB wil100.
Strategy:
1. Precompute row sums of A and B for fast O(n) move delta evaluation.
2. Generate a strong initial solution using a greedy rank-matching heuristic.
3. Perform Tabu Search:
   - Evaluate all O(n^2) swap moves.
   - Select the best non-tabu move (or best move if all are tabu but better than current).
   - Maintain a tabu list of recently swapped positions to prevent cycling.
   - Include an aspiration criterion: accept a tabu move if it improves the best known solution.
4. If time permits, perform a few diversification restarts (perturbing the best solution) 
   and re-run tabu search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return np.sum(a * b[np.ix_(p, p)])

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

def _tabu_search(a, b, p_init, time_limit, tabu_tenure=10, seed=None):
    """
    Perform Tabu Search starting from p_init.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    n = len(p_init)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute row sums for delta calculation
    # Delta for swapping i and j:
    # Let pi = p[i], pj = p[j]
    # Change in cost = 
    #   (a[i,j] + a[j,i]) * (b[pi,pj] - b[pj,pi])  -- mutual term change? No.
    # Let's derive properly.
    # Cost = sum_{k,l} a[k,l] b[p[k],p[l]]
    # Swap i, j.
    # Terms involving i or j change.
    # 
    # Let's use the vectorized delta formula derived earlier but optimized.
    # delta = sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    #       + sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    #       + a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
    #       + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    
    # This is O(n) per move. With n=100, 100*99/2 ~ 5000 moves.
    # 5000 * 100 ops = 500k ops per iteration.
    # 100 iterations = 50M ops. Very fast in numpy.
    
    start_time = time.time()
    tabu_list = {} # position -> iteration when it became tabu
    current_iter = 0
    
    # We will evaluate all moves each iteration to find the best one.
    # This is a "best improvement" tabu search.
    
    while time.time() - start_time < time_limit:
        best_move_i = -1
        best_move_j = -1
        best_move_cost = np.inf
        best_move_delta = 0
        
        # Evaluate all pairs (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                pi = p[i]
                pj = p[j]
                
                # Check tabu status
                # A move is tabu if either i or j was swapped in the last 'tabu_tenure' iterations
                # and the resulting cost does not beat the best known cost (aspiration)
                is_tabu = False
                if i in tabu_list and current_iter - tabu_list[i] < tabu_tenure:
                    is_tabu = True
                if j in tabu_list and current_iter - tabu_list[j] < tabu_tenure:
                    is_tabu = True
                
                # Calculate delta
                # Optimization: Pre-compute vectors? 
                # For fixed i, j:
                # term1 = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
                # term2 = sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
                # term3 = a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
                # term4 = a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
                
                # Vectorized calculation for terms 1 and 2
                # Create mask for k != i, j
                # This is slow if done in a loop. 
                # Let's try to vectorize the inner loop over j for a fixed i? 
                # No, b[p[k], pj] depends on j.
                
                # Let's just compute it. N=100 is small.
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                p_masked = p[mask]
                
                # term1: sum over k in mask of (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
                # b[pj, p_masked] and b[pi, p_masked] are vectors of length n-2
                diff_a_1 = a[i, mask] - a[j, mask]
                diff_b_1 = b[pj, p_masked] - b[pi, p_masked]
                term1 = np.sum(diff_a_1 * diff_b_1)
                
                # term2: sum over k in mask of (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
                diff_a_2 = a[mask, i] - a[mask, j]
                diff_b_2 = b[p_masked, pj] - b[p_masked, pi]
                term2 = np.sum(diff_a_2 * diff_b_2)
                
                term3 = a[i, i] * (b[pj, pj] - b[pi, pi]) + a[j, j] * (b[pi, pi] - b[pj, pj])
                term4 = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
                
                delta = term1 + term2 + term3 + term4
                new_cost = current_cost + delta
                
                # Aspiration criterion: if new_cost < best_cost, it's allowed even if tabu
                if is_tabu and new_cost >= best_cost:
                    continue
                
                if new_cost < best_move_cost:
                    best_move_cost = new_cost
                    best_move_delta = delta
                    best_move_i = i
                    best_move_j = j
        
        # If no valid move found (should not happen usually), break
        if best_move_i == -1:
            break
        
        # Perform the move
        p[best_move_i], p[best_move_j] = p[best_move_j], p[best_move_i]
        current_cost = best_move_cost
        
        # Update tabu list
        tabu_list[best_move_i] = current_iter
        tabu_list[best_move_j] = current_iter
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        current_iter += 1
        
        # Clean up old tabu entries (optional, for memory, but n is small)
        if current_iter > 100:
            # Remove entries older than tenure
            keys_to_remove = [k for k, v in tabu_list.items() if current_iter - v >= tabu_tenure]
            for k in keys_to_remove:
                del tabu_list[k]
        
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    # Deadline: 120s total. Leave 5s for overhead/safety.
    total_deadline = 115.0
    start_time = time.time()
    
    # 1. Initial Solution
    p_init = _greedy_initial(a, b)
    best_p = p_init
    best_cost = _qap_cost(a, b, p_init)
    
    # 2. Main Tabu Search
    # Allocate 80% of time to the first deep tabu search
    time_for_main = total_deadline * 0.8
    p_ls, cost_ls = _tabu_search(a, b, p_init, time_for_main, tabu_tenure=10)
    
    if cost_ls < best_cost:
        best_p = p_ls
        best_cost = cost_ls
    
    # 3. Diversification / Restart
    # If time remains, try a few restarts from perturbed best solution
    time_remaining = total_deadline - (time.time() - start_time)
    
    if time_remaining > 5:
        # Perturb best solution with a few random swaps
        num_restarts = 3
        time_per_restart = (time_remaining - 2) / num_restarts
        
        for r in range(num_restarts):
            if time.time() - start_time > total_deadline - 1:
                break
                
            p_restart = best_p.copy()
            # Perform 5-10 random swaps to diversify
            num_swaps = 5 + r * 2
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
                
            # Run tabu search from this perturbed state
            p_rs, cost_rs = _tabu_search(a, b, p_restart, time_per_restart, tabu_tenure=10)
            
            if cost_rs < best_cost:
                best_p = p_rs
                best_cost = cost_rs
                
    return best_p.tolist()

# EVOLVE-BLOCK-END
