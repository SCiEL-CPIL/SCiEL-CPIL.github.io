# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    p_arr = np.asarray(p)
    return np.sum(a * b[np.ix_(p_arr, p_arr)])


def _compute_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    p_arr = np.asarray(p)
    return np.sum(a * b[np.ix_(p_arr, p_arr)])


def _delta_swap(a, b, p, i, j):
    """Compute the delta cost for swapping positions i and j in permutation p."""
    li = p[i]
    lj = p[j]
    
    # Precompute vectors for faster computation
    # delta = sum_{k != i,j} [a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + 
    #                          a[k,i]*(b[p[k],lj] - b[p[k],li]) +
    #                          a[j,k]*(b[li,p[k]] - b[lj,p[k]]) +
    #                          a[k,j]*(b[p[k],li] - b[p[k],lj])]
    #       + a[i,i]*(b[lj,lj] - b[li,li]) + a[j,j]*(b[li,li] - b[lj,lj])
    #       + a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li])
    
    # Vectorized computation
    pk = np.delete(p, [i, j])  # positions of all other facilities
    
    # Terms involving i and k
    term_ik = a[i, :] * (b[lj, :] - b[li, :])
    term_ki = a[:, i] * (b[:, lj] - b[:, li])
    
    # Terms involving j and k
    term_jk = a[j, :] * (b[li, :] - b[lj, :])
    term_kj = a[:, j] * (b[:, li] - b[:, lj])
    
    # Sum over all k except i and j
    delta = np.sum(term_ik[pk]) + np.sum(term_ki[pk]) + np.sum(term_jk[pk]) + np.sum(term_kj[pk])
    
    # Add diagonal and i-j terms
    delta += a[i, i] * (b[lj, lj] - b[li, li])
    delta += a[j, j] * (b[li, li] - b[lj, lj])
    delta += a[i, j] * (b[lj, li] - b[li, lj])
    delta += a[j, i] * (b[li, lj] - b[lj, li])
    
    return delta


def _greedy_initial(a, b, n):
    """Create a good initial solution using a greedy heuristic."""
    # Strategy: Start with empty assignment, add one facility at a time
    # to the location that minimizes the incremental cost.
    
    # Precompute row sums of a and b for quick estimates
    a_row_sums = a.sum(axis=1)
    
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
    
    return assigned.copy().tolist()


def _local_search(a, b, p, time_limit=100):
    """Improve the solution using 2-opt style local search with best improvement."""
    p = list(p)
    n = len(p)
    start_time = time.time()
    current_cost = _qap_cost(a, b, p)
    
    best_cost = current_cost
    best_p = p[:]
    
    # Precompute for faster evaluation
    # We'll use a vectorized approach to evaluate all swaps at once
    
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        # Try all pairs and find the best improvement
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Use random sampling for large n to keep it fast
        if n > 100:
            # Sample a subset of pairs
            num_samples = min(5000, n * n // 4)
            for _ in range(num_samples):
                i, j = random.sample(range(n), 2)
                delta = _delta_swap(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        else:
            # Try all pairs for smaller n
            for i in range(n):
                for j in range(i + 1, n):
                    delta = _delta_swap(a, b, p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
        
        if best_delta < 0 and best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p[:]
        else:
            # No improvement found, break
            break
    
    return best_p, best_cost


def _perturb(p, n, strength=10, seed=None):
    """Perturb the solution by performing random swaps."""
    p = list(p)
    rng = random.Random(seed)
    for _ in range(strength):
        i, j = rng.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Get a good initial solution using greedy
    p = _greedy_initial(a, b, n)
    
    # Run local search to improve
    p, cost = _local_search(a, b, p, time_limit=min(30, time_limit - (time.time() - start_time)))
    
    best_p = p[:]
    best_cost = cost
    
    # Iterated Local Search
    rng = random.Random(42)
    perturbation_strength = 5
    
    while time.time() - start_time < time_limit:
        # Perturb the solution
        p_perturbed = _perturb(best_p, n, strength=perturbation_strength, seed=rng.randint(0, 2**31))
        
        # Run local search on perturbed solution
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_perturbed, time_limit=min(remaining_time - 2, 20))
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls[:]
            # Reset perturbation strength on improvement
            perturbation_strength = 5
        else:
            # Increase perturbation strength if stuck
            perturbation_strength = min(perturbation_strength + 1, 20)
    
    return best_p
# EVOLVE-BLOCK-END
