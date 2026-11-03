"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the QAP cost for assignment p (vectorized)."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _greedy_initial(a, b, rng):
    """Greedy construction with random restarts: build assignment one facility at a time.
    For each facility (in random order), assign it to the location that minimizes the
    marginal cost given already-assigned facilities."""
    n = len(a)
    order = rng.permutation(n)
    p = np.full(n, -1, dtype=np.int64)
    # We'll track which locations are used
    used = np.zeros(n, dtype=bool)
    # Precompute row sums of a for potential use, but we do incremental marginal cost
    # marginal cost of placing facility i at location j given current partial assignment:
    # sum over already-assigned facilities k: a[i,k]*b[j,p[k]] + a[k,i]*b[p[k],j]
    # We compute this incrementally.
    
    # For efficiency, maintain an array `marg` of shape (n,) where marg[j] = 
    # sum over assigned k of a[i,k]*b[j,p[k]] + a[k,i]*b[p[k],j] ... but this depends on i.
    # Actually, for a given i, the marginal cost for location j is:
    #   sum_{k assigned} a[i,k] * b[j, p[k]] + a[k,i] * b[p[k], j]
    # Let's precompute for each i, as we go.
    
    # Alternative: just compute it on the fly for each candidate location.
    # With n=256, and 256 facilities, each step checks up to 256 locations,
    # and for each location we sum over up to 256 assigned facilities. That's O(n^3) = 256^3 ~ 16M ops, fine.
    
    assigned_set = []
    for i in order:
        best_loc = -1
        best_cost = np.inf
        for j in range(n):
            if used[j]:
                continue
            # Compute marginal cost of placing i at j
            cost = 0.0
            for k in assigned_set:
                cost += a[i, k] * b[j, p[k]] + a[k, i] * b[p[k], j]
            if cost < best_cost:
                best_cost = cost
                best_loc = j
        p[i] = best_loc
        used[best_loc] = True
        assigned_set.append(i)
    return p


def _local_search_2opt(a, b, p, max_iters=5000, deadline=None, safety_margin=5.0):
    """2-opt local search: try swapping pairs of locations. Vectorized evaluation."""
    n = len(a)
    # Precompute the current cost
    bp = b[np.ix_(p, p)]
    current_cost = int(np.sum(a * bp))
    
    # For a swap of p[i] and p[j], the change in cost can be computed.
    # Let's define: for each position i, the contribution to cost from row i and column i of a
    # times the corresponding entries in b.
    
    # Actually, let's think about the delta for swapping locations p[i] and p[j].
    # Current: a[i,j] contributes a[i,j]*b[p[i],p[j]] and a[j,i]*b[p[j],p[i]]
    # After swap: a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
    # Also, for any other k, a[i,k]*b[p[i],p[k]] becomes a[i,k]*b[p[j],p[k]] and vice versa.
    
    # This is complex. Let's use a different approach: for each pair (i,j), compute the 
    # delta by updating the cost. With n=256, there are ~32k pairs. For each swap, we can
    # compute the delta in O(n) time using precomputed row/col contributions.
    
    # Let's define for each facility i and location l:
    #   row_contrib[i, l] = sum_k a[i,k] * b[l, p[k]]  (if i is placed at l, contribution from row i)
    #   col_contrib[i, l] = sum_k a[k,i] * b[p[k], l]  (if i is placed at l, contribution from col i)
    # Total cost = sum_i (row_contrib[i, p[i]] + col_contrib[i, p[i]]) / 2 ... no, that double counts.
    
    # Actually, cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = sum_i sum_j a[i,j] * b[p[i], p[j]]
    
    # For a swap of positions i and j (i.e., swap p[i] and p[j]):
    # The new cost = old cost - (terms involving i or j with old positions) + (terms with new positions)
    
    # Let me just do a simpler approach: for each candidate swap, compute the delta in O(n).
    # With 32k pairs and O(n)=256 per pair, that's 32k * 256 = 8M operations per full sweep.
    # With 5000 iterations, that's 40B ops... too slow.
    
    # Better: precompute the "interaction" matrix and use it to compute deltas faster.
    
    # Let's define: for the current assignment p, define
    #   R[i, l] = sum_k a[i,k] * b[l, p[k]]  for all i, l
    #   C[i, l] = sum_k a[k,i] * b[p[k], l]  for all i, l
    # Then cost = sum_i (R[i, p[i]] + C[i, p[i]]) / 2? No.
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # For fixed i: sum_j a[i,j] * b[p[i], p[j]] = R[i, p[i]]
    # So cost = sum_i R[i, p[i]]
    
    # If we swap p[i] and p[j] (let u=p[i], v=p[j]):
    # New R[i, v] = sum_k a[i,k] * b[v, p'[k]] where p' has p'[i]=v, p'[j]=u, others same.
    # R[i, v] = a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
    # Old R[i, u] = a[i,i]*b[u,u] + a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u,p[k]]
    
    # This is getting complicated. Let's just use a randomized approach with limited swaps.
    
    # Simpler: do random swaps and keep if they improve.
    best_p = p.copy()
    best_cost = current_cost
    start_time = time.time()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - safety_margin:
            break
        
        i, j = rng.choice(n, 2, replace=False)
        if i == j:
            continue
        
        # Swap
        p[i], p[j] = p[j], p[i]
        
        # Compute new cost
        bp_new = b[np.ix_(p, p)]
        new_cost = int(np.sum(a * bp_new))
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = p.copy()
        else:
            # Revert
            p[i], p[j] = p[j], p[i]
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate multiple greedy initial solutions
    best_p = None
    best_cost = np.inf
    
    # Do several greedy constructions
    n_greedy = 5
    for g in range(n_greedy):
        if time.time() > deadline - 20.0:
            break
        p_g = _greedy_initial(a, b, rng)
        cost_g = _qap_cost(a, b, p_g)
        if cost_g < best_cost:
            best_cost = cost_g
            best_p = p_g
    
    # Local search from the best greedy solution
    if time.time() < deadline - 10.0:
        best_p, best_cost = _local_search_2opt(a, b, best_p, max_iters=2000, deadline=deadline, safety_margin=5.0, rng=rng)
    
    # If time remains, try more local search iterations or additional restarts
    if time.time() < deadline - 5.0:
        # Do another round of local search with more iterations
        best_p, best_cost = _local_search_2opt(a, b, best_p, max_iters=5000, deadline=deadline, safety_margin=3.0, rng=rng)
    
    return best_p.tolist()


# Fix: _local_search_2opt needs rng parameter
def _local_search_2opt(a, b, p, max_iters=5000, deadline=None, safety_margin=5.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(a)
    bp = b[np.ix_(p, p)]
    current_cost = int(np.sum(a * bp))
    
    best_p = p.copy()
    best_cost = current_cost
    start_time = time.time()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - safety_margin:
            break
        
        i, j = rng.choice(n, 2, replace=False)
        
        p[i], p[j] = p[j], p[i]
        bp_new = b[np.ix_(p, p)]
        new_cost = int(np.sum(a * bp_new))
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = p.copy()
        else:
            p[i], p[j] = p[j], p[i]
    
    return best_p, best_cost
# EVOLVE-BLOCK-END
