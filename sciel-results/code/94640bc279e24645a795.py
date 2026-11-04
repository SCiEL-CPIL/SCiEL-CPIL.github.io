"""
Tabu Search for QAPLIB wil100 with O(n) exact delta computation.
Strategy:
1. Greedy construction using precomputed row sums for a fast, decent initial solution.
2. Tabu Search with:
   - Exact O(n) cost delta calculation for 2-swap moves.
   - Short-term memory (tabu list) to prevent cycling.
   - Aspiration criterion to override tabu status if a move improves the global best.
   - Diversification: If no improvement for a while, apply a penalty to frequently used moves or restart from a perturbed best.
3. Final deterministic steepest-descent local search to polish the best solution found.
"""
import numpy as np
import time
from collections import deque

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _init_greedy(a, b):
    """
    Greedy initialization:
    1. Compute row sums of a (facility flows) and b (location distances).
    2. Sort facilities by flow (descending) and locations by distance (ascending) - 
       heavy flows should go to central locations.
    3. Assign in that order.
    This is a heuristic that often lands in a good basin.
    """
    n = len(a)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # We want to minimize sum a_sums[i] * b_sums[p[i]] roughly.
    # Sort facilities by high flow, locations by low distance.
    fac_order = np.argsort(a_sums)[::-1]
    loc_order = np.argsort(b_sums)
    
    p = np.empty(n, dtype=np.int64)
    # p[i] = location. 
    # For facility fac_order[k], assign location loc_order[k].
    for k in range(n):
        f = fac_order[k]
        l = loc_order[k]
        p[f] = l
        
    return p

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the exact change in cost for swapping facilities at positions i and j.
    p: current permutation (p[k] is location of facility k).
    i, j: indices of facilities to swap.
    
    Cost C(p) = sum_{u,v} a[u,v] * b[p[u], p[v]]
    
    When swapping i and j:
    p_new[i] = p[j], p_new[j] = p[i], others same.
    
    Delta = C(p_new) - C(p)
    
    Terms affected involve indices i or j in u or v.
    1. u=i, v=i: a[i,i] * (b[p[j],p[j]] - b[p[i],p[i]])
    2. u=j, v=j: a[j,j] * (b[p[i],p[i]] - b[p[j],p[j]])
    3. u=i, v=j: a[i,j] * (b[p[j],p[i]] - b[p[i],p[j]])
    4. u=j, v=i: a[j,i] * (b[p[i],p[j]] - b[p[j],p[i]])
    5. u=i, v=k (k!=i,j): a[i,k] * (b[p[j],p[k]] - b[p[i],p[k]])
    6. u=j, v=k (k!=i,j): a[j,k] * (b[p[i],p[k]] - b[p[j],p[k]])
    7. u=k, v=i (k!=i,j): a[k,i] * (b[p[k],p[j]] - b[p[k],p[i]])
    8. u=k, v=j (k!=i,j): a[k,j] * (b[p[k],p[i]] - b[p[k],p[j]])
    
    We can vectorize the sums over k.
    """
    pi = p[i]
    pj = p[j]
    
    # Diagonal terms
    delta = 0
    delta += a[i,i] * (b[pj,pj] - b[pi,pi])
    delta += a[j,j] * (b[pi,pi] - b[pj,pj])
    delta += a[i,j] * (b[pj,pi] - b[pi,pj])
    delta += a[j,i] * (b[pi,pj] - b[pj,pi])
    
    # Off-diagonal terms involving i or j and k != i,j
    # Sum over k != i,j
    # We can compute sum over all k and subtract k=i and k=j terms if they were included, 
    # but it's easier to just mask or use the fact that we already handled i,j pairs.
    
    # Term 5: sum_{k != i,j} a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
    # Term 6: sum_{k != i,j} a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
    # Term 7: sum_{k != i,j} a[k,i] * (b[p[k], pj] - b[p[k], pi])
    # Term 8: sum_{k != i,j} a[k,j] * (b[p[k], pi] - b[p[k], pj])
    
    # Let's compute sums over all k first, then subtract k=i and k=j contributions if needed?
    # Actually, the terms above are for k != i,j.
    # Let S5_all = sum_{k} a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
    # S5_all includes k=i and k=j.
    # k=i: a[i,i] * (b[pj, pi] - b[pi, pi])
    # k=j: a[i,j] * (b[pj, pj] - b[pi, pj])
    
    # This is getting complicated to subtract manually. 
    # Let's just use numpy to compute the full sums and subtract the specific i,j terms.
    
    # Vectorized computation for terms 5-8
    # p is (n,). p[k] is location of facility k.
    
    # b_pj_p: (n,) where b_pj_p[k] = b[pj, p[k]]
    # b_pi_p: (n,) where b_pi_p[k] = b[pi, p[k]]
    # b_p_pj: (n,) where b_p_pj[k] = b[p[k], pj]
    # b_p_pi: (n,) where b_p_pi[k] = b[p[k], pi]
    
    b_pj_p = b[pj, p]
    b_pi_p = b[pi, p]
    b_p_pj = b[p, pj]
    b_p_pi = b[p, pi]
    
    # Sum 5: sum_k a[i,k] * (b_pj_p[k] - b_pi_p[k])
    sum5 = np.dot(a[i, :], b_pj_p - b_pi_p)
    # Subtract k=i and k=j
    sum5 -= a[i,i] * (b[pj, pi] - b[pi, pi])
    sum5 -= a[i,j] * (b[pj, pj] - b[pi, pj])
    
    # Sum 6: sum_k a[j,k] * (b_pi_p[k] - b_pj_p[k])
    sum6 = np.dot(a[j, :], b_pi_p - b_pj_p)
    sum6 -= a[j,j] * (b[pi, pj] - b[pj, pj])
    sum6 -= a[j,i] * (b[pi, pi] - b[pj, pi])
    
    # Sum 7: sum_k a[k,i] * (b_p_pj[k] - b_p_pi[k])
    sum7 = np.dot(a[:, i], b_p_pj - b_p_pi)
    sum7 -= a[i,i] * (b[pi, pj] - b[pi, pi])
    sum7 -= a[j,i] * (b[pj, pj] - b[pj, pi])
    
    # Sum 8: sum_k a[k,j] * (b_p_pi[k] - b_p_pj[k])
    sum8 = np.dot(a[:, j], b_p_pi - b_p_pj)
    sum8 -= a[i,j] * (b[pi, pi] - b[pi, pj])
    sum8 -= a[j,j] * (b[pj, pi] - b[pj, pj])
    
    delta += sum5 + sum6 + sum7 + sum8
    
    return int(delta)

def _tabu_search(a, b, p_init, time_limit, start_time):
    n = len(p_init)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Tabu list: store (i, j, time_excluded)
    # To be efficient, we can store just the pair and a counter.
    # Or use a set of frozensets? For n=100, number of pairs is 4950.
    # A simple list of tuples is fine.
    tabu_list = deque()
    tabu_tenure = 20 # Number of iterations a move is tabu
    
    # Frequency matrix for diversification
    freq = np.zeros((n, n), dtype=np.int32)
    
    no_improve_count = 0
    max_no_improve = 50
    
    # Diversification strategy: if stuck, apply penalty to frequent moves
    # Or just restart from best with perturbation.
    
    while time.time() - start_time < time_limit:
        # Evaluate all possible 2-swaps? n=100 -> 4950 swaps.
        # O(n) per swap -> 4950 * 100 = 495,000 ops. Very fast in numpy?
        # Actually, _compute_swap_delta uses np.dot which is O(n).
        # 4950 calls to a function with O(n) numpy ops might be slow in Python loop.
        # Let's vectorize the evaluation of all swaps?
        # That's hard because p changes.
        # But we can evaluate a subset or use a smarter approach.
        
        # Given n=100, 4950 swaps is small.
        # Let's try to evaluate all of them.
        
        best_delta = np.inf
        best_i, best_j = -1, -1
        is_tabu = False
        aspiration = False
        
        # To speed up, we can precompute some vectors?
        # For each pair (i,j), delta depends on p[i], p[j].
        
        # Let's iterate.
        for i in range(n):
            for j in range(i+1, n):
                # Check tabu
                # We need an efficient way to check if (i,j) is tabu.
                # A set of active tabu moves?
                # Let's maintain a set of active tabu pairs.
                pass
        
        # This nested loop in Python is too slow for 120s if we do it every iteration.
        # 4950 iterations * ~100 ops = 500k ops. Python overhead is ~1us per op?
        # 500k * 1us = 0.5s per iteration. 120s / 0.5s = 240 iterations.
        # That's very few iterations.
        
        # We need a faster evaluation.
        # Can we vectorize the delta calculation for all pairs?
        # Delta(i,j) = f(p[i], p[j], i, j)
        # It involves sums over k.
        
        # Alternative: Use the vectorized batch delta from the previous program?
        # The previous program used O(n^2) per swap? No, it was O(n^2) for the batch of m swaps?
        # Wait, the previous program's _compute_batch_deltas was O(m*n^2) or O(m*n)?
        # It did: diff_a_row = a[i_idx, :] - a[j_idx, :] # (m, n)
        # Then T1 = np.sum(diff_a_row * diff_b_col, axis=1) # (m,)
        # This is O(m*n).
        # If m=4950, n=100, it's 495,000 ops.
        # This is vectorized in NumPy!
        
        # Let's use a vectorized approach to evaluate all 4950 swaps at once.
        
        # Generate all pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        m = len(i_idx)
        
        # Compute deltas for all pairs
        # We need to adapt _compute_batch_deltas to handle this.
        # The previous _compute_batch_deltas assumed i_arr, j_arr are the swaps.
        # It computed:
        # pi = p[i_idx]
        # pj = p[j_idx]
        # diff_a_row = a[i_idx, :] - a[j_idx, :] # (m, n)
        # ...
        # This is O(m*n).
        
        # Let's implement a vectorized all-swaps delta evaluator.
        
        deltas = _compute_all_swap_deltas(a, b, p, i_idx, j_idx)
        
        # Apply tabu constraints
        # We need to know which of these m swaps are tabu.
        # Maintain a boolean array `is_tabu` of size m?
        # Or a set of (i,j) pairs?
        # A set lookup for 4950 items is fast.
        
        # Let's maintain a set of active tabu pairs.
        # And an aspiration check: if delta < best_cost - current_cost (improvement), allow it.
        
        # For now, let's just pick the best non-tabu move, or best move if aspiration.
        
        # To avoid O(m) python loop for tabu check, let's keep a list of active tabu pairs
        # and filter the indices.
        
        # Actually, simpler:
        # 1. Find indices of improving moves (deltas < 0).
        # 2. If none, find best move overall.
        # 3. Check tabu status.
        
        # Let's just iterate over the best few candidates?
        # Or, since m=4950, we can just loop in Python to check tabu and aspiration.
        # 4950 iterations is fast in Python (~0.1ms).
        
        best_valid_idx = -1
        best_valid_delta = np.inf
        
        for k in range(m):
            d = deltas[k]
            i = i_idx[k]
            j = j_idx[k]
            
            # Aspiration: if this move improves the global best, it's allowed even if tabu
            # Note: d is delta from current. So new_cost = current_cost + d.
            # Aspiration if new_cost < best_cost.
            aspiration = (current_cost + d < best_cost)
            
            # Tabu check
            # We need a fast way to check if (i,j) is in tabu_list.
            # Let's use a set of active tabu pairs.
            # We'll update the set at the end of the loop.
            
            if (i, j) in active_tabu_set and not aspiration:
                continue
                
            if d < best_valid_delta:
                best_valid_delta = d
                best_valid_idx = k
        
        if best_valid_idx == -1:
            # No valid move found? Should not happen unless all are tabu and no aspiration.
            # Break or force a random move.
            break
            
        i = i_idx[best_valid_idx]
        j = j_idx[best_valid_idx]
        
        # Apply move
        p[i], p[j] = p[j], p[i]
        current_cost += best_valid_delta
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update tabu list
        # Add (i,j) to tabu
        active_tabu_set.add((i, j))
        tabu_list.append((i, j, time.time())) # Store time to expire
        
        # Expire old tabu moves
        while tabu_list:
            ti, tj, t_time = tabu_list[0]
            if time.time() - t_time > 0.01: # 10ms tenure? Or use iteration count.
                tabu_list.popleft()
                active_tabu_set.discard((ti, tj))
            else:
                break
        
        # Diversification: if no improve for long, perturb
        if no_improve_count > max_no_improve:
            # Perturb best solution
            p = best_p.copy()
            current_cost = best_cost
            # Apply 5 random swaps
            for _ in range(5):
                ri = np.random.randint(n)
                rj = np.random.randint(n)
                if ri != rj:
                    p[ri], p[rj] = p[rj], p[ri]
                    current_cost += _compute_swap_delta(a, b, p, ri, rj)
            no_improve_count = 0
            # Clear tabu list
            tabu_list.clear()
            active_tabu_set.clear()

    return best_p, best_cost

def _compute_all_swap_deltas(a, b, p, i_idx, j_idx):
    """
    Vectorized computation of delta costs for a batch of swaps.
    i_idx, j_idx: arrays of facility indices to swap.
    Returns array of delta costs.
    """
    m = len(i_idx)
    n = len(p)
    
    pi = p[i_idx]
    pj = p[j_idx]
    
    # (m, n) arrays
    # diff_a_row[m, k] = a[i, k] - a[j, k]
    diff_a_row = a[i_idx, :] - a[j_idx, :]
    
    # diff_a_col[m, k] = a[k, i] - a[k, j]
    diff_a_col = (a[:, i_idx] - a[:, j_idx]).T
    
    # diff_b_col[m, k] = b[pj, p[k]] - b[pi, p[k]]
    b_pj_p = b[np.ix_(pj, p)]
    b_pi_p = b[np.ix_(pi, p)]
    diff_b_col = b_pj_p - b_pi_p
    
    # diff_b_row[m, k] = b[p[k], pj] - b[p[k], pi]
    b_p_pj = b[np.ix_(p, pj)].T
    b_p_pi = b[np.ix_(p, pi)].T
    diff_b_row = b_p_pj - b_p_pi
    
    # T1 = sum_k diff_a_row * diff_b_col
    T1 = np.sum(diff_a_row * diff_b_col, axis=1)
    
    # T2 = sum_k diff_a_col * diff_b_row
    T2 = np.sum(diff_a_col * diff_b_row, axis=1)
    
    # Corrections
    corr_i_T1 = (a[i_idx, i_idx] - a[j_idx, i_idx]) * (b[pj, pi] - b[pi, pi])
    corr_i_T2 = (a[i_idx, i_idx] - a[i_idx, j_idx]) * (b[pi, pj] - b[pi, pi])
    
    corr_j_T1 = (a[i_idx, j_idx] - a[j_idx, j_idx]) * (b[pj, pj] - b[pi, pj])
    corr_j_T2 = (a[j_idx, i_idx] - a[j_idx, j_idx]) * (b[pj, pj] - b[pj, pi])
    
    direct1 = (a[i_idx, j_idx] - a[j_idx, i_idx]) * (b[pj, pi] - b[pi, pj])
    direct2 = a[i_idx, i_idx] * (b[pj, pj] - b[pi, pi])
    direct3 = a[j_idx, j_idx] * (b[pi, pi] - b[pj, pj])
    
    deltas = T1 + T2 - corr_i_T1 - corr_i_T2 - corr_j_T1 - corr_j_T2 + direct1 + direct2 + direct3
    
    return deltas.astype(np.int64)

def _final_polish(a, b, p, time_limit, start_time):
    """
    Deterministic steepest descent local search.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    while time.time() - start_time < time_limit:
        deltas = _compute_all_swap_deltas(a, b, p, i_idx, j_idx)
        
        if np.min(deltas) >= 0:
            break
            
        best_idx = np.argmin(deltas)
        i = i_idx[best_idx]
        j = j_idx[best_idx]
        
        p[i], p[j] = p[j], p[i]
        current_cost += deltas[best_idx]
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 145.0
    start_time = time.time()
    
    np.random.seed(42)
    
    # Phase 1: Greedy Init
    p_init = _init_greedy(a, b)
    
    # Phase 2: Tabu Search
    elapsed = time.time() - start_time
    ts_time = min(100.0, total_deadline - elapsed - 20)
    if ts_time > 5.0:
        best_p, best_cost = _tabu_search(a, b, p_init, ts_time, start_time)
    else:
        best_p = p_init
        best_cost = _qap_cost(a, b, p_init)
        
    # Phase 3: Final Polish
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 3:
        ls_time = total_deadline - elapsed - 2
        p_final, cost_final = _final_polish(a, b, best_p.copy(), ls_time, start_time)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
            
    return best_p.tolist()
