"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _calc_row_sums(a, b):
    """Precompute row sums for fast delta calculations."""
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    return a_row_sums, b_row_sums


def _delta_2swap(a, b, p, i, j, a_row_sums, b_row_sums):
    """
    Compute cost delta for swapping positions i and j in permutation p.
    p maps facility -> location. So p[i] is the location of facility i.
    
    Actually, let's be precise:
    Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    
    Swapping p[i] and p[j] changes:
    - Terms involving (i, i), (i, j), (j, i), (j, j)
    - Terms involving (k, i) and (k, j) for all k != i, j
    - Terms involving (i, k) and (j, k) for all k != i, j
    """
    pi = p[i]
    pj = p[j]
    
    # Old contribution of pairs involving i or j
    # We need to compute the change carefully.
    
    # Let's use the standard formula for 2-swap delta.
    # Delta = -2 * sum_{k != i,j} (a[i,k]*(b[pj,k] - b[pi,k]) + a[j,k]*(b[pi,k] - b[pj,k]))
    #        + a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
    #        + (a[i,j] + a[j,i])*(b[pj,pi] - b[pi,pj])
    
    # More precisely, let's derive it:
    # Old terms involving i or j:
    # sum_k a[i,k]*b[pi,p[k]] + sum_k a[j,k]*b[pj,p[k]] 
    #   - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi]  (to avoid double counting, but actually the sum includes k=i and k=j)
    
    # Let's just compute it directly for the affected terms.
    # Affected terms: (i,k) for all k, (j,k) for all k, (k,i) for all k, (k,j) for all k
    # But (i,i), (i,j), (j,i), (j,j) are counted twice if we're not careful.
    
    # Simpler approach: compute old and new contributions of all terms where at least one index is i or j.
    
    # Old:
    # sum_{k} a[i,k]*b[pi,p[k]] + sum_{k} a[k,i]*b[p[k],pi] 
    #   - a[i,i]*b[pi,pi]  (counted twice, subtract once)
    # + sum_{k} a[j,k]*b[pj,p[k]] + sum_{k} a[k,j]*b[p[k],pj]
    #   - a[j,j]*b[pj,pj]  (counted twice, subtract once)
    # - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi]  (these are in both sums, subtract once)
    
    # This is getting complicated. Let's use a cleaner formulation.
    
    # The total cost can be written as:
    # C = sum_{i,j} a[i,j]*b[p[i],p[j]]
    
    # When we swap p[i] and p[j], let's call the new permutation p'.
    # p'[i] = pj, p'[j] = pi, p'[k] = p[k] for k != i,j
    
    # Delta = C' - C
    
    # Let's compute the change term by term.
    # Terms not involving i or j: unchanged.
    # Terms involving i or j:
    
    # (i,i): a[i,i]*(b[pj,pj] - b[pi,pi])
    # (j,j): a[j,j]*(b[pi,pi] - b[pj,pj])
    # (i,j): a[i,j]*(b[pj,pi] - b[pi,pj])
    # (j,i): a[j,i]*(b[pi,pj] - b[pj,pi])
    
    # For k != i,j:
    # (i,k): a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
    # (k,i): a[k,i]*(b[p[k],pj] - b[p[k],pi])
    # (j,k): a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    # (k,j): a[k,j]*(b[p[k],pi] - b[p[k],pj])
    
    # Sum these up.
    
    # Let's vectorize.
    # Create a mask for k != i, j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.where(mask)[0]
    
    if len(ks) > 0:
        pk = p[ks]
        # (i,k) terms: a[i,ks] * (b[pj, pk] - b[pi, pk])
        delta_ik = np.sum(a[i, ks] * (b[pj, pk] - b[pi, pk]))
        # (k,i) terms: a[ks, i] * (b[pk, pj] - b[pk, pi])
        delta_ki = np.sum(a[ks, i] * (b[pk, pj] - b[pk, pi]))
        # (j,k) terms: a[j, ks] * (b[pi, pk] - b[pj, pk])
        delta_jk = np.sum(a[j, ks] * (b[pi, pk] - b[pj, pk]))
        # (k,j) terms: a[ks, j] * (b[pk, pi] - b[pk, pj])
        delta_kj = np.sum(a[ks, j] * (b[pk, pi] - b[pk, pj]))
    else:
        delta_ik = delta_ki = delta_jk = delta_kj = 0
    
    delta_ii = a[i, i] * (b[pj, pj] - b[pi, pi])
    delta_jj = a[j, j] * (b[pi, pi] - b[pj, pj])
    delta_ij = a[i, j] * (b[pj, pi] - b[pi, pj])
    delta_ji = a[j, i] * (b[pi, pj] - b[pj, pi])
    
    return delta_ii + delta_jj + delta_ij + delta_ji + delta_ik + delta_ki + delta_jk + delta_kj


def _local_search_2swap(a, b, p, max_iters=500, time_limit=None, start_time=None):
    """Perform 2-swap local search. Returns improved permutation and cost."""
    n = len(p)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    p = p.copy()
    current_cost = _compute_cost(a, b, p)
    
    if time_limit is None:
        time_limit = 120.0
    if start_time is None:
        start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() - start_time > time_limit - 5:
            break
        
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Sample a subset of swaps for speed
        if n > 50:
            # Sample random pairs
            num_samples = min(500, n * (n - 1) // 2)
            for _ in range(num_samples):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                delta = _delta_2swap(a, b, p, i, j, a_row_sums, b_row_sums)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    delta = _delta_2swap(a, b, p, i, j, a_row_sums, b_row_sums)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
    
    return p, current_cost


def _or_opt_move(a, b, p, start, length, new_pos):
    """
    Compute delta for moving a block of facilities from [start, start+length)
    to position new_pos.
    """
    n = len(p)
    
    # Identify the block
    block = list(range(start, start + length))
    
    # Compute old and new costs for affected terms
    # This is complex; for simplicity, let's just compute the full cost change
    # by constructing the new permutation and computing the cost difference
    # for the affected facilities.
    
    # For now, let's implement a simpler version: try all possible block moves
    # and compute the delta by direct cost computation of affected terms.
    
    # Actually, let's implement a proper delta calculation.
    # When we move a block, the positions of the block elements change,
    # and the positions of elements between old and new positions shift.
    
    # This is getting complex. Let's use a different approach:
    # Implement Or-opt by trying to move a single facility (length=1) to a new position,
    # which is a special case of 2-swap but allows inserting anywhere.
    
    # For length=1, moving facility at position start to new_pos is equivalent to
    # a sequence of adjacent swaps, but we can compute the delta directly.
    
    # Let's implement a general block move delta.
    
    # For now, let's keep it simple and just do 2-swap local search.
    # The Or-opt idea can be approximated by trying to insert a facility at a new position.
    
    return None


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Multi-start with local search
    best_p = p.copy()
    best_cost = _compute_cost(a, b, p)
    
    # Do a few random restarts with local search
    num_restarts = 5
    random.seed(42)
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Create a perturbed starting solution
        if restart == 0:
            p_curr = best_p.copy()
        else:
            p_curr = best_p.copy()
            # Random perturbation: swap a few random pairs
            num_swaps = random.randint(2, min(10, n // 5))
            for _ in range(num_swaps):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
        
        # Local search
        p_curr, cost_curr = _local_search_2swap(
            a, b, p_curr, 
            max_iters=200, 
            time_limit=time_limit,
            start_time=start_time
        )
        
        if cost_curr < best_cost:
            best_cost = cost_curr
            best_p = p_curr.copy()
    
    # Final polish with more iterations
    if time.time() - start_time < time_limit - 5:
        best_p, best_cost = _local_search_2swap(
            a, b, best_p,
            max_iters=500,
            time_limit=time_limit,
            start_time=start_time
        )
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
