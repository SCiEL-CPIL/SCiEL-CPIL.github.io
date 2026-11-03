"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    # sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient computation:
    # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # We can compute this as:
    # For each pair (i,j), the contribution is a[i,j] * b[p[i], p[j]]
    # 
    # A faster way:
    # Let c[i,j] = b[p[i], p[j]]
    # Then cost = sum_{i,j} a[i,j] * c[i,j]
    # 
    # But we can also use:
    # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b[p[i], :]) ... no that's not right
    # 
    # Actually: sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # For fixed i, sum_j a[i,j] * b[p[i], p[j]] = (a[i,:] * b[p[i], :]).sum()
    # So total = sum_i (a[i,:] * b[p[i], :]).sum()
    
    # Even faster: precompute
    # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i] * b[p[i]]).sum()
    
    p_arr = np.asarray(p)
    total = 0
    for i in range(len(a)):
        total += np.dot(a[i], b[p_arr[i]])
    return total


def _greedy_initial(a, b, n):
    """Create a good initial solution using a greedy heuristic."""
    # Strategy: Start with empty assignment, add one facility at a time
    # to the location that minimizes the incremental cost.
    
    # Precompute row sums of a and b for quick estimates
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)  # b is symmetric presumably, so row sums = col sums
    
    # We'll build the permutation incrementally
    # assigned[i] = location assigned to facility i, or -1 if unassigned
    assigned = np.full(n, -1, dtype=np.int64)
    used_locations = set()
    
    # Sort facilities by their "importance" (row sum of a) in descending order
    facility_order = np.argsort(-a_row_sums, kind='stable')
    
    for idx in facility_order:
        best_loc = -1
        best_cost = np.inf
        
        for loc in range(n):
            if loc in used_locations:
                continue
            
            # Compute incremental cost of assigning facility idx to location loc
            # The incremental cost is:
            # sum over already-assigned facilities k of:
            #   a[idx, k] * b[loc, assigned[k]] + a[k, idx] * b[assigned[k], loc]
            # Plus the self-term: a[idx, idx] * b[loc, loc] (usually 0)
            
            inc_cost = a[idx, idx] * b[loc, loc]
            
            for k in range(n):
                if assigned[k] != -1:
                    inc_cost += a[idx, k] * b[loc, assigned[k]]
                    inc_cost += a[k, idx] * b[assigned[k], loc]
            
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_loc = loc
        
        assigned[idx] = best_loc
        used_locations.add(best_loc)
    
    # Convert to permutation p where p[i] = location of facility i
    p = assigned.copy()
    return p.tolist()


def _local_search_2opt(a, b, p, time_limit=100):
    """Improve the solution using 2-opt style local search (swap two assignments)."""
    p = list(p)
    n = len(p)
    start_time = time.time()
    current_cost = _qap_cost(a, b, p)
    
    # Precompute for faster incremental cost evaluation
    # When we swap p[i] and p[j], the change in cost can be computed efficiently.
    
    # Let's use a simpler approach: try all pairs and evaluate the delta
    # For swapping facilities i and j (i.e., swap p[i] and p[j]):
    # The change in cost involves terms where i or j are involved.
    
    # Let loc_i = p[i], loc_j = p[j]
    # Before: terms involving i or j in the sum sum_{u,v} a[u,v] * b[p[u], p[v]]
    # After swap: p'[i] = loc_j, p'[j] = loc_i, others unchanged.
    
    # Delta = sum_{u,v} a[u,v] * (b[p'[u], p'[v]] - b[p[u], p[v]])
    # Only terms where u in {i,j} or v in {i,j} change.
    
    # Let's compute delta for swap of i and j:
    # Let li = p[i], lj = p[j]
    # 
    # Terms that change:
    # 1. (i,i): a[i,i]*(b[lj,lj] - b[li,li])
    # 2. (j,j): a[j,j]*(b[li,li] - b[lj,lj])
    # 3. (i,k) for k not in {i,j}: a[i,k]*(b[lj, p[k]] - b[li, p[k]])
    # 4. (k,i) for k not in {i,j}: a[k,i]*(b[p[k], lj] - b[p[k], li])
    # 5. (j,k) for k not in {i,j}: a[j,k]*(b[li, p[k]] - b[lj, p[k]])
    # 6. (k,j) for k not in {i,j}: a[k,j]*(b[p[k], li] - b[p[k], lj])
    # 7. (i,j): a[i,j]*(b[lj, li] - b[li, lj])
    # 8. (j,i): a[j,i]*(b[li, lj] - b[lj, li])
    
    # This is O(n) per pair evaluation, so O(n^3) total which is too slow for n=256.
    # 256^3 = 16.7M, might be borderline. Let's try to be smarter.
    
    # Alternative: Use a smarter neighborhood or limit the number of iterations.
    
    # For n=256, let's try a limited number of swaps with random restarts.
    
    best_cost = current_cost
    best_p = p[:]
    
    # Try random swaps
    random.seed(42)
    
    max_iterations = 50000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() - start_time > time_limit:
            break
        
        # Pick two random distinct facilities
        i, j = random.sample(range(n), 2)
        
        li = p[i]
        lj = p[j]
        
        # Compute delta for swapping p[i] and p[j]
        # We need to compute the change in total cost.
        
        # Let's compute it directly but efficiently.
        # delta = sum_{u,v} a[u,v] * (b[p_new[u], p_new[v]] - b[p_old[u], p_old[v]])
        # where p_new differs from p_old only at positions i and j.
        
        # The affected terms are those where u in {i,j} or v in {i,j}.
        
        # Let's compute:
        # old_contrib = sum over affected (u,v) of a[u,v]*b[p[u],p[v]]
        # new_contrib = sum over affected (u,v) of a[u,v]*b[p_new[u],p_new[v]]
        
        # Affected pairs: (i,i), (i,j), (i,k) for k!=i,j, (j,i), (j,j), (j,k) for k!=i,j,
        #                 (k,i) for k!=i,j, (k,j) for k!=i,j
        
        # Let's compute delta directly:
        delta = 0
        
        # (i,i)
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        # (j,j)
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        # (i,j)
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        # (j,i)
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        # (i,k) for k not in {i,j}
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            delta += a[i, k] * (b[lj, pk] - b[li, pk])
            delta += a[j, k] * (b[li, pk] - b[lj, pk])
            delta += a[k, i] * (b[pk, lj] - b[pk, li])
            delta += a[k, j] * (b[pk, li] - b[pk, lj])
        
        new_cost = current_cost + delta
        
        if new_cost < current_cost:
            # Accept the swap
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            iteration += 1
        else:
            iteration += 1
        
        # Early termination if no improvement for a while
        if iteration % 1000 == 0 and current_cost == best_cost:
            # Check if we've been stuck
            pass
    
    return p, current_cost


def solve(a, b):
    n = len(a)
    
    # Get a good initial solution using greedy
    p = _greedy_initial(a, b, n)
    
    # Run local search to improve
    p, cost = _local_search_2opt(a, b, p, time_limit=100)
    
    return p
# EVOLVE-BLOCK-END
