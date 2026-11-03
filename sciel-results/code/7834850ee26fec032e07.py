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
    
    # T1: column part - sum_k a[k,i]*b[p[k],pj] - a[k,j]*b[p[k],pi] corrections
    # Use the identity: delta = sum_k [a[k,i]*b[pj,p[k]] - a[k,i]*b[pi,p[k]] - a[k,j]*b[pj,p[k]] + a[k,j]*b[pi,p[k]]]
    #                       + sum_k [a[i,k]*b[pj,p[k]] - a[i,k]*b[pi,p[k]] - a[j,k]*b[pj,p[k]] + a[j,k]*b[pi,p[k]]]
    #                       - 2*(above terms for k=i,j to avoid double counting)
    # Simpler: just compute directly with dot products
    
    # For k not in {i,j}: contribution from a[k,i] and a[k,j] with b[pj,*] and b[pi,*]
    # T1 = sum_k a[k,i]*(b[pj,p[k]] - b[pi,p[k]]) - a[k,j]*(b[pj,p[k]] - b[pi,p[k]])
    #    = sum_k (a[k,i] - a[k,j]) * (b[pj,p[k]] - b[pi,p[k]])
    
    diff_a_col = a[:, i] - a[:, j]  # shape (n,)
    diff_b_row = b[p, pj] - b[p, pi]  # shape (n,)
    T1 = np.dot(diff_a_col, diff_b_row)
    # Correct for k=i: a[i,i] should pair with b[pi,pi] not b[pj,pi]
    # In the sum above, for k=i: diff_a_col[i] * diff_b_row[i] = (a[i,i]-a[i,j])*(b[pj,pi]-b[pi,pi])
    # But actual contribution for k=i when i moves to pj: a[i,i]*b[pj,pi]... 
    # Actually the formula is correct as-is for the "other" positions. Let me use the standard formula.
    
    # Standard swap delta formula:
    # delta = (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
    #       + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[pj,p[k]]-b[pi,p[k]])
    #       + sum_{k!=i,j} (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
    
    # T1 = sum_k (a[k,i]-a[k,j])*(b[pj,p[k]]-b[pi,p[k]]) includes k=i,j
    # T2 = sum_k (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]]) includes k=i,j
    # T3 = cross term
    
    # The full expression is T1 + T2 + T3 where T3 handles the i<->j interaction
    # Let me just compute it directly:
    
    # T1: column contributions (excluding i,j handled separately)
    # For all k: (a[k,i]-a[k,j]) * (b[pj,p[k]] - b[pi,p[k]])
    # This includes k=i and k=j which need correction
    
    # For k=i: (a[i,i]-a[i,j]) * (b[pj,pi] - b[pi,pi])  -- this is part of T1
    # For k=j: (a[j,i]-a[j,j]) * (b[pj,pj] - b[pi,pj])  -- this is part of T1
    
    # T2: row contributions
    # For all k: (a[i,k]-a[j,k]) * (b[pj,p[k]] - b[pi,p[k]])
    # For k=i: (a[i,i]-a[j,i]) * (b[pj,pi] - b[pi,pi])
    # For k=j: (a[i,j]-a[j,j]) * (b[pj,pj] - b[pi,pj])
    
    # T3: the interaction between i and j themselves
    # When we swap, a[i,j] pairs with b[pi,pj] and a[j,i] pairs with b[pj,pi]
    # Before: a[i,j] paired with b[pi,pj]... wait no.
    
    # Let me use the proven formula from the original:
    T1 = np.dot(diff_a_col, diff_b_row)
    corr_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    corr_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = T1 - corr_i - corr_j
    
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    T2 = np.dot(diff_a_row, diff_b_col)
    corr2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    corr2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = T2 - corr2_i - corr2_j
    
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return int(T1 + T2 + T3)

def _local_search(a, b, p, current_cost, max_iters=2000, time_limit=5.0):
    """
    Fast local search using precomputed row/column contributions.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    # Precompute for fast delta evaluation
    # col_contrib[i] = sum_k a[k,i] * b[p[i], p[k]] for the column part
    # row_contrib[i] = sum_k a[i,k] * b[p[i], p[k]] for the row part
    
    # We'll use a different approach: precompute b_p = b[p, :] (rows of b indexed by p)
    # and use matrix operations
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Fast vectorized computation of all swap deltas
        # For each pair (i,j), delta = (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
        #   + sum_k!=i,j (a[k,i]-a[k,j])*(b[pj,pk]-b[pi,pk])
        #   + sum_k!=i,j (a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk])
        
        # Vectorize over i for fixed j, or use broadcasting
        # This is O(n^2 * n) = O(n^3) which is 10^6 for n=100, doable in numpy
        
        # Precompute diff_a_cols: for each i, a[:,i] - a[:,j] varies with j
        # Better: compute all deltas at once using matrix ops
        
        # deltas[i,j] = (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
        #              + sum_k (a[k,i]-a[k,j])*(b[p[j],p[k]]-b[p[i],p[k]])
        #              + sum_k (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
        
        # Let B_p = b[p,:] so B_p[i,k] = b[p[i],p[k]]
        B_p = b[np.ix_(p, p)]  # B_p[i,k] = b[p[i], p[k]]
        
        # Term 1: (a[i,j]-a[j,i]) * (B_p[j,i] - B_p[i,j])
        # Note B_p is symmetric if b is symmetric, so B_p[j,i] = B_p[i,j], making term 1 zero
        # But b may not be symmetric, so keep it
        
        # Term 2: sum_k (a[k,i]-a[k,j]) * (B_p[j,k] - B_p[i,k])
        # = sum_k a[k,i]*B_p[j,k] - a[k,i]*B_p[i,k] - a[k,j]*B_p[j,k] + a[k,j]*B_p[i,k]
        # = (a[:,i] @ B_p[j,:]) - (a[:,i] @ B_p[i,:]) - (a[:,j] @ B_p[j,:]) + (a[:,j] @ B_p[i,:])
        
        # Term 3: sum_k (a[i,k]-a[j,k]) * (B_p[j,k] - B_p[i,k])
        # = (a[i,:] @ B_p[j,:]) - (a[i,:] @ B_p[i,:]) - (a[j,:] @ B_p[j,:]) + (a[j,:] @ B_p[i,:])
        
        # So delta[i,j] = (a[i,j]-a[j,i])*(B_p[j,i]-B_p[i,j])
        #                + a[:,i]@(B_p[j,:]-B_p[i,:]) - a[:,j]@(B_p[j,:]-B_p[i,:])
        #                + a[i,:]@(B_p[j,:]-B_p[i,:]) - a[j,:]@(B_p[j,:]-B_p[i,:])
        
        # = (a[i,j]-a[j,i])*(B_p[j,i]-B_p[i,j])
        #   + (a[:,i]-a[:,j]) @ (B_p[j,:]-B_p[i,:])
        #   + (a[i,:]-a[j,:]) @ (B_p[j,:]-B_p[i,:])
        
        # Let diff_a_col[i,j] = a[:,i] - a[:,j] (n x n x n) - too big
        # Let diff_a_row[i,j] = a[i,:] - a[j,:] (n x n x n) - too big
        # Let diff_B[i,j] = B_p[j,:] - B_p[i,:] (n x n x n) - too big
        
        # Alternative: loop over i, vectorize over j
        for i in range(n):
            # diff_a_col_i = a[:, i] - a[:, j] for all j -> shape (n, n)
            # diff_a_row_i = a[i, :] - a[j, :] for all j -> shape (n, n)
            # diff_B_i = B_p[j, :] - B_p[i, :] for all j -> shape (n, n)
            
            diff_a_col_i = a[:, i][np.newaxis, :] - a  # (n, n): diff_a_col_i[j, k] = a[k,i] - a[k,j]
            # Wait, a[:,i] is shape (n,), a is shape (n,n)
            # a[:,i][np.newaxis,:] - a gives (n,n) where [j,k] = a[k,i] - a[k,j]
            
            diff_a_row_i = a[i, :][np.newaxis, :] - a.T  # (n, n): [j,k] = a[i,k] - a[j,k]
            # a[i,:] is shape (n,), a.T is shape (n,n)
            # a[i,:][np.newaxis,:] - a.T gives [j,k] = a[i,k] - a[j,k]
            
            diff_B_i = B_p - B_p[i, :][np.newaxis, :]  # (n, n): [j,k] = B_p[j,k] - B_p[i,k]
            
            # term2[j] = diff_a_col_i[j,:] @ diff_B_i[j,:]
            term2 = np.einsum('jk,jk->j', diff_a_col_i, diff_B_i)
            
            # term3[j] = diff_a_row_i[j,:] @ diff_B_i[j,:]
            term3 = np.einsum('jk,jk->j', diff_a_row_i, diff_B_i)
            
            # term1[j] = (a[i,j] - a[j,i]) * (B_p[j,i] - B_p[i,j])
            term1 = (a[i, :] - a[:, i]) * (B_p[:, i] - B_p[i, :])
            
            deltas_i = term1 + term2 + term3
            
            # Only consider j > i
            for j in range(i + 1, n):
                delta = int(deltas_i[j])
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
    Uses 2-opt style moves (swap two random positions) with adaptive temperature.
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
    
    # Track no-improvement counter for adaptive restart
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < duration:
        # Generate a random swap
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
            surrogate_delta = (a_row_sums[i] * b_col_sums[p[j]] + a_row_sums[j] * b_col_sums[p[i]]
                              - a_row_sums[i] * b_col_sums[p[i]] - a_row_sums[j] * b_col_sums[p[j]])
            
            if surrogate_delta < 0 and delta > 0:
                accept_prob = np.exp(-delta / (T * 0.5))
            elif surrogate_delta > 0 and delta < 0:
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
                no_improve = 0
            else:
                no_improve += 1
        
        # Adaptive: if stuck for too long, do a quick random restart
        if no_improve > max_no_improve:
            # Perturb: do 10 random swaps
            for _ in range(10):
                i2 = np.random.randint(n)
                j2 = np.random.randint(n)
                if i2 != j2:
                    p[i2], p[j2] = p[j2], p[i2]
            cost = _qap_cost(a, b, p)
            no_improve = 0
    
    return best_p, best_cost

def _tabu_search(a, b, p, current_cost, time_limit=5.0, tabu_tenure=15):
    """
    Fast tabu search using a set-based tabu structure for O(1) lookup.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    # Tabu set: stores frozenset({i,j}) with expiry
    tabu_expiries = {}  # frozenset({i,j}) -> expiry iteration
    iteration = 0
    max_iterations = 2000
    
    while time.time() - start_time < time_limit and iteration < max_iterations:
        # Clean expired tabu entries
        expired_keys = [k for k, exp in tabu_expiries.items() if exp <= iteration]
        for k in expired_keys:
            del tabu_expiries[k]
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Use the fast vectorized approach for delta computation
        B_p = b[np.ix_(p, p)]
        
        for i in range(n):
            diff_a_col_i = a[:, i][np.newaxis, :] - a  # (n, n)
            diff_a_row_i = a[i, :][np.newaxis, :] - a.T  # (n, n)
            diff_B_i = B_p - B_p[i, :][np.newaxis, :]  # (n, n)
            
            term2 = np.einsum('jk,jk->j', diff_a_col_i, diff_B_i)
            term3 = np.einsum('jk,jk->j', diff_a_row_i, diff_B_i)
            term1 = (a[i, :] - a[:, i]) * (B_p[:, i] - B_p[i, :])
            
            deltas_i = term1 + term2 + term3
            
            for j in range(i + 1, n):
                # Check tabu
                key = frozenset((i, j))
                if key in tabu_expiries:
                    continue
                
                delta = int(deltas_i[j])
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break  # No non-tabu improving move found
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta
        
        # Add to tabu
        key = frozenset((best_i, best_j))
        tabu_expiries[key] = iteration + tabu_tenure
        iteration += 1
        
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
    
    # Phase 2: SA runs with adaptive time allocation
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0  # Reserve 20s for final polish
    
    if remaining <= 0:
        return best_p.tolist()
    
    # Run 1: From best initial, with surrogate feedback
    seg1_time = remaining * 0.40
    T_start = 500.0
    T_end = 5.0
    p1, cost1 = _sa_run(a, b, best_p, best_cost, seg1_time, T_start, T_end, use_surrogate_feedback=True)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = p1.copy()
    
    # Run 2: From second-best initial, standard SA
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0
    if remaining > 0:
        seg2_time = remaining * 0.40
        T_start = 300.0
        T_end = 3.0
        p2, cost2 = _sa_run(a, b, candidates[1][0], candidates[1][1], seg2_time, T_start, T_end, use_surrogate_feedback=False)
        if cost2 < best_cost:
            best_cost = cost2
            best_p = p2.copy()
    
    # Run 3: From third-best initial, standard SA
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0
    if remaining > 0:
        seg3_time = remaining * 0.40
        T_start = 200.0
        T_end = 2.0
        p3, cost3 = _sa_run(a, b, candidates[2][0], candidates[2][1], seg3_time, T_start, T_end, use_surrogate_feedback=False)
        if cost3 < best_cost:
            best_cost = cost3
            best_p = p3.copy()
    
    # Phase 3: Final polish - local search then tabu search
    elapsed = time.time() - start_time
    polish_budget = max(0.0, total_deadline - elapsed - 5.0)
    
    if polish_budget > 1.0:
        # First: fast local search
        ls_time = polish_budget * 0.5
        best_p, best_cost = _local_search(a, b, best_p, best_cost, max_iters=10000, time_limit=ls_time)
        
        # Then: tabu search to escape any remaining local minimum
        elapsed = time.time() - start_time
        tabu_time = max(0.0, total_deadline - elapsed - 5.0)
        if tabu_time > 1.0:
            best_p, best_cost = _tabu_search(a, b, best_p, best_cost, time_limit=tabu_time, tabu_tenure=15)
    
    # Safety check: ensure we return within deadline
    elapsed = time.time() - start_time
    if elapsed > total_deadline - 3.0:
        # We're close to the deadline, return immediately
        pass
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
