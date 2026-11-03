"""
Simulated Annealing solver for QAPLIB wil100.
Uses Benders-inspired decomposition: a linearized surrogate guides initial
assignment, while exact quadratic cost feedback tightens the search.
Multi-start SA with adaptive cooling and final local polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _linear_surrogate_cost(a, b, p):
    """
    Benders master: linearized surrogate cost.
    Uses row sums of A and column sums of B (for symmetric B, same as row sums).
    surrogate = sum_i (a_row_sum[i]) * (b_col_sum[p[i]])
    This is a relaxation that ignores the quadratic interaction.
    """
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    return int(np.sum(a_row_sums * b_col_sums[p]))

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

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns the integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    # Vectorized computation of all swap deltas at once
    # For swapping positions i and j:
    # delta = sum_k!=i,j a[k,i]*b[p[k],pj] + a[k,j]*b[p[k],pi] - a[k,i]*b[p[k],pi] - a[k,j]*b[p[k],pj]
    #       + a[i,k]*b[pj,p[k]] + a[j,k]*b[pi,p[k]] - a[i,k]*b[pi,p[k]] - a[j,k]*b[pj,p[k]]
    #       + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] - a[i,i]*b[pi,pi] - a[j,j]*b[pj,pj]
    #       + a[i,i]*b[pj,pj] + a[j,j]*b[pi,pi] - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi]
    
    # Simplified: use the standard QAP swap delta formula
    # Let's compute it efficiently using precomputed structures
    
    pi_val = p[i]
    pj_val = p[j]
    
    # All other positions
    other = np.arange(len(p))
    mask = (other != i) & (other != j)
    k = other[mask]
    
    # Column part: for each k in other, a[k,i]*b[p[k],pj] - a[k,i]*b[p[k],pi] + a[k,j]*b[p[k],pi] - a[k,j]*b[p[k],pj]
    col_part = np.dot(a[k, i] - a[k, j], b[p[k], pj_val] - b[p[k], pi_val])
    
    # Row part: for each k in other, a[i,k]*b[pj,p[k]] - a[i,k]*b[pi,p[k]] + a[j,k]*b[pi,p[k]] - a[j,k]*b[pj,p[k]]
    row_part = np.dot(a[i, k] - a[j, k], b[pj_val, p[k]] - b[pi_val, p[k]])
    
    # Self and cross terms
    self_cross = (a[i, j] * b[pj_val, pi_val] + a[j, i] * b[pi_val, pj_val]
                  - a[i, i] * b[pi_val, pi_val] - a[j, j] * b[pj_val, pj_val]
                  + a[i, i] * b[pj_val, pj_val] + a[j, j] * b[pi_val, pi_val]
                  - a[i, j] * b[pi_val, pj_val] - a[j, i] * b[pj_val, pi_val])
    
    return int(col_part + row_part + self_cross)

def _compute_all_swap_deltas(a, b, p):
    """
    Vectorized computation of all swap deltas for permutation p.
    Returns a 2D array of shape (n, n) where entry [i,j] is the delta for swapping i and j.
    """
    n = len(p)
    deltas = np.zeros((n, n), dtype=np.int64)
    
    # Precompute
    a_rows = a  # a[i, :]
    b_cols = b  # b[:, p[i]]
    
    for i in range(n):
        for j in range(i + 1, n):
            deltas[i, j] = _compute_swap_delta(a, b, p, i, j)
            deltas[j, i] = deltas[i, j]
    
    return deltas

def _local_search(a, b, p, current_cost, max_iters=2000, time_limit=5.0):
    """
    Steepest-descent local search: repeatedly find the best improving swap.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Compute all swap deltas
        for i in range(n):
            for j in range(i + 1, n):
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break  # Local minimum
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta
    
    return p, cost

def _sa_run(a, b, p_init, current_cost, duration, T_start, T_end, use_surrogate_feedback=False):
    """
    Run simulated annealing for a given duration.
    If use_surrogate_feedback, adjust acceptance based on surrogate cost gap.
    Returns (best_p, best_cost).
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    # Precompute for surrogate feedback
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        # Benders-inspired: if we have surrogate feedback, use it to modulate acceptance
        if use_surrogate_feedback and T > 10.0:
            # O(1) surrogate delta: only positions i and j change
            # surrogate = sum_k a_row_sums[k] * b_col_sums[p[k]]
            # delta_surrogate = a_row_sums[i]*b_col_sums[p[j]] + a_row_sums[j]*b_col_sums[p[i]]
            #                  - a_row_sums[i]*b_col_sums[p[i]] - a_row_sums[j]*b_col_sums[p[j]]
            surrogate_delta = (a_row_sums[i] * b_col_sums[p[j]] + a_row_sums[j] * b_col_sums[p[i]]
                              - a_row_sums[i] * b_col_sums[p[i]] - a_row_sums[j] * b_col_sums[p[j]])
            
            # If surrogate improves but exact cost worsens, be more accepting
            # (the quadratic terms may be compensating)
            if surrogate_delta < 0 and delta > 0:
                # Boost acceptance probability
                accept_prob = np.exp(-delta / (T * 0.5))  # More permissive
            elif surrogate_delta > 0 and delta < 0:
                # Surrogate worsens but exact improves - definitely accept
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / T)
        else:
            accept_prob = np.exp(-delta / T)
        
        if delta < 0 or np.random.rand() < accept_prob:
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Generate multiple initial solutions
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_random1 = _random_initial(n)
    cost_random1 = _qap_cost(a, b, p_random1)
    
    p_random2 = _random_initial(n)
    cost_random2 = _qap_cost(a, b, p_random2)
    
    # Select the best initial solution
    candidates = [(p_greedy, cost_greedy), (p_random1, cost_random1), (p_random2, cost_random2)]
    candidates.sort(key=lambda x: x[1])
    best_p, best_cost = candidates[0]
    best_p = best_p.copy()
    
    # Phase 2: SA with Benders-inspired surrogate feedback
    # Split remaining time into 3 segments for 3 SA runs with different starting points
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 10.0  # Reserve 10s for final polish
    
    # Run 1: From best initial, with surrogate feedback
    seg1_time = remaining * 0.4
    T_start = 500.0
    T_end = 5.0
    p1, cost1 = _sa_run(a, b, best_p, best_cost, seg1_time, T_start, T_end, use_surrogate_feedback=True)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = p1.copy()
    
    # Run 2: From second-best initial, standard SA
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 10.0
    seg2_time = remaining * 0.5
    T_start = 300.0
    T_end = 3.0
    p2, cost2 = _sa_run(a, b, candidates[1][0], candidates[1][1], seg2_time, T_start, T_end, use_surrogate_feedback=False)
    if cost2 < best_cost:
        best_cost = cost2
        best_p = p2.copy()
    
    # Run 3: From third-best initial, standard SA
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 10.0
    seg3_time = remaining * 0.6
    T_start = 200.0
    T_end = 2.0
    p3, cost3 = _sa_run(a, b, candidates[2][0], candidates[2][1], seg3_time, T_start, T_end, use_surrogate_feedback=False)
    if cost3 < best_cost:
        best_cost = cost3
        best_p = p3.copy()
    
    # Phase 3: Final local search polish
    elapsed = time.time() - start_time
    polish_time = min(8.0, max(0.0, total_deadline - elapsed - 2.0))
    if polish_time > 0.5:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, max_iters=5000, time_limit=polish_time)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
