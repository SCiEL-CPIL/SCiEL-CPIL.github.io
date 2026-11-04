"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    deadline = time.time() + 118.0
    
    def compute_cost(p):
        # p is a permutation of locations for facilities
        # cost = sum a[i, j] * b[p[i], p[j]]
        # Using vectorization for speed
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        """
        Compute delta cost for swapping facilities i and j.
        p is the current assignment.
        """
        u = p[i]
        v = p[j]
        
        # Vectorized calculation of the difference
        # Terms involving i:
        # Old: sum_k a[i, k] b[u, p[k]] + sum_k a[k, i] b[p[k], u]
        # New: sum_k a[i, k] b[v, p[k]] + sum_k a[k, i] b[p[k], v]
        # Diff_i = sum_k a[i, k] (b[v, p[k]] - b[u, p[k]]) + sum_k a[k, i] (b[p[k], v] - b[p[k], u])
        
        # Terms involving j:
        # Old: sum_k a[j, k] b[v, p[k]] + sum_k a[k, j] b[p[k], v]
        # New: sum_k a[j, k] b[u, p[k]] + sum_k a[k, j] b[p[k], u]
        # Diff_j = sum_k a[j, k] (b[u, p[k]] - b[v, p[k]]) + sum_k a[k, j] (b[p[k], u] - b[p[k], v])
        
        # Direct terms for i, j interaction and self terms:
        # Old: a[i, j] b[u, v] + a[j, i] b[v, u] + a[i, i] b[u, u] + a[j, j] b[v, v]
        # New: a[i, j] b[v, u] + a[j, i] b[u, v] + a[i, i] b[v, v] + a[j, j] b[u, u]
        # Note: The sums above include k=i and k=j. We need to be careful not to double count or miss corrections.
        
        # Let's compute the raw sums first (including i and j)
        # Sum_i_raw = dot(a[i, :], b[v, p] - b[u, p]) + dot(a[:, i], b[p, v] - b[p, u])
        # Sum_j_raw = dot(a[j, :], b[u, p] - b[v, p]) + dot(a[:, j], b[p, u] - b[p, v])
        
        diff_b_vu = b[v, p] - b[u, p]
        diff_b_uv = b[u, p] - b[v, p]
        diff_b_pv = b[p, v] - b[p, u]
        diff_b_pu = b[p, u] - b[p, v]
        
        sum_i = np.dot(a[i, :], diff_b_vu) + np.dot(a[:, i], diff_b_pv)
        sum_j = np.dot(a[j, :], diff_b_uv) + np.dot(a[:, j], diff_b_pu)
        
        # The sums sum_i and sum_j include the terms where k=i and k=j.
        # We need to subtract the "old" contributions of k=i and k=j from these sums 
        # and add the "new" contributions? 
        # Actually, the standard derivation is:
        # Delta = (New Cost) - (Old Cost)
        
        # Let's use the explicit correction method which is robust.
        # Sum_i includes:
        # k=i: a[i, i] (b[v, u] - b[u, u]) + a[i, i] (b[u, v] - b[u, u]) -- Wait, b[p[i], v] is b[u, v].
        # Let's stick to the pre-computed subtractions from the previous successful version but optimized.
        
        # Subtractions for i (terms where k=i or k=j in the sums)
        # In sum_i (dot with diff_b_vu and diff_b_pv):
        # k=i term in first dot: a[i, i] * (b[v, p[i]] - b[u, p[i]]) = a[i, i] * (b[v, u] - b[u, u])
        # k=i term in second dot: a[i, i] * (b[p[i], v] - b[p[i], u]) = a[i, i] * (b[u, v] - b[u, u])
        # k=j term in first dot: a[i, j] * (b[v, p[j]] - b[u, p[j]]) = a[i, j] * (b[v, v] - b[u, v])
        # k=j term in second dot: a[j, i] * (b[p[j], v] - b[p[j], u]) = a[j, i] * (b[v, v] - b[v, u])
        
        sub_i = a[i, i] * (b[v, u] - b[u, u]) + a[i, i] * (b[u, v] - b[u, u]) + \
                a[i, j] * (b[v, v] - b[u, v]) + a[j, i] * (b[v, v] - b[v, u])
        
        # In sum_j:
        # k=i term in first dot: a[j, i] * (b[u, p[i]] - b[v, p[i]]) = a[j, i] * (b[u, u] - b[v, u])
        # k=i term in second dot: a[i, j] * (b[p[i], u] - b[p[i], v]) = a[i, j] * (b[u, u] - b[u, v])
        # k=j term in first dot: a[j, j] * (b[u, p[j]] - b[v, p[j]]) = a[j, j] * (b[u, v] - b[v, v])
        # k=j term in second dot: a[j, j] * (b[p[j], u] - b[p[j], v]) = a[j, j] * (b[v, u] - b[v, v])
        
        sub_j = a[j, i] * (b[u, u] - b[v, u]) + a[i, j] * (b[u, u] - b[u, v]) + \
                a[j, j] * (b[u, v] - b[v, v]) + a[j, j] * (b[v, u] - b[v, v])
        
        # Direct interaction change (i, j) and self (i, i), (j, j)
        # Old: a[i, j] b[u, v] + a[j, i] b[v, u] + a[i, i] b[u, u] + a[j, j] b[v, v]
        # New: a[i, j] b[v, u] + a[j, i] b[u, v] + a[i, i] b[v, v] + a[j, j] b[u, u]
        direct = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u]) + \
                 a[i, i] * (b[v, v] - b[u, u]) + a[j, j] * (b[u, u] - b[v, v])
        
        return (sum_i - sub_i) + (sum_j - sub_j) + direct

    def local_search_2opt(p, current_cost, deadline):
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Steepest descent
            for i in range(n - 1):
                if time.time() >= deadline - 2.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    d = swap_delta(i, j, p)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                
        return p, current_cost

    def local_search_3opt(p, current_cost, deadline):
        # Try 3-cycles: i -> j -> k -> i
        # This is expensive O(n^3). We will only use it sparingly or on small subsets.
        # For n=150, n^3 = 3.3M. One full pass is feasible if optimized, but swap_delta is O(n).
        # So O(n^4) total. 150^4 = 500M. Too slow for full pass.
        # We will skip full 3-opt and rely on 2-opt + perturbations.
        # Instead, we implement a "3-swap" move which is just 3 random swaps or specific patterns.
        # Actually, let's stick to 2-opt but improve the perturbation.
        return p, current_cost

    # Initialization: Greedy Construction
    # We want to build p sequentially.
    # Cost of adding facility i to location l, given current assignment P:
    # Delta = sum_{k in P} a[i, k] b[l, p[k]] + sum_{k in P} a[k, i] b[p[k], l] + a[i, i] b[l, l]
    
    def greedy_init(seed=0):
        rng = np.random.RandomState(seed)
        unassigned_facs = list(range(n))
        unassigned_locs = list(range(n))
        p = np.full(n, -1, dtype=np.int64)
        
        # Pre-compute some sums to speed up? 
        # For each candidate location l, we need to evaluate cost against all assigned facilities.
        # This is O(n) per candidate, O(n^2) per step, O(n^3) total.
        # 150^3 = 3.3M operations. Very fast.
        
        for step in range(n):
            if time.time() > deadline - 10.0:
                break
                
            best_loc = -1
            best_cost = np.inf
            
            # If we have few facilities left, we can be more exhaustive or just pick best.
            # For speed, we evaluate all remaining locations.
            for l in unassigned_locs:
                # Calculate incremental cost
                inc_cost = 0
                # We need to sum over assigned facilities k
                # Assigned facilities are those where p[k] != -1
                # We can maintain a list of assigned facilities
                
                # Optimization:
                # Let A be the set of assigned facilities.
                # Cost = sum_{k in A} a[i, k] b[l, p[k]] + sum_{k in A} a[k, i] b[p[k], l] + a[i, i] b[l, l]
                
                # We can vectorize this if we keep track of assigned facilities.
                # But simple loop is fine for n=150.
                
                # To avoid iterating over all n facilities and checking if assigned,
                # we maintain a list of assigned facilities.
                
                # Let's just iterate over all k in range(n) and check if assigned? 
                # No, that's O(n) per l, so O(n^2) per step.
                
                # Better: maintain `assigned_facs` list.
                
                # For now, let's just do the full sum over assigned_facs.
                
                # We need to know which facilities are assigned.
                # Let's just loop over all k in range(n) and check p[k] != -1.
                # This is 150 * 150 * 150 = 3.3M ops.
                
                for k in range(n):
                    if p[k] != -1:
                        loc_k = p[k]
                        inc_cost += a[i, k] * b[l, loc_k]
                        inc_cost += a[k, i] * b[loc_k, l]
                
                inc_cost += a[i, i] * b[l, l]
                
                if inc_cost < best_cost:
                    best_cost = inc_cost
                    best_loc = l
            
            if best_loc == -1:
                # Fallback if time ran out or something
                best_loc = unassigned_locs[0]
            
            p[i] = best_loc
            unassigned_locs.remove(best_loc)
            
        return p

    # Better Greedy: Use a linearization for the "future" cost to guide the choice?
    # The simple greedy above is myopic.
    # Let's try a "Linear Greedy" where we estimate the cost of placing i at l
    # assuming the remaining facilities are placed optimally (approximated by average).
    
    def linear_greedy_init(seed=0):
        rng = np.random.RandomState(seed)
        unassigned_facs = list(range(n))
        unassigned_locs = list(range(n))
        p = np.full(n, -1, dtype=np.int64)
        
        # Pre-compute row/col sums of b to estimate average distance?
        # Or use the matrix product C = a @ b.
        # C[i, l] = sum_k a[i, k] b[l, k].
        # This represents the cost of placing i at l if k is distributed according to the "average" location of k.
        
        C = a @ b
        
        for step in range(n):
            if time.time() > deadline - 10.0:
                break
                
            best_loc = -1
            best_cost = np.inf
            
            # For each candidate location l for facility i:
            # 1. Immediate cost with assigned neighbors
            # 2. Estimated future cost: C[i, l] * (fraction of remaining flow?)
            
            # Let's combine:
            # Real cost with assigned:
            # Est cost with unassigned:
            # We don't know which unassigned facilities will be where.
            # A simple heuristic: Minimize (Immediate Cost + C[i, l])
            
            for l in unassigned_locs:
                inc_cost = 0
                for k in range(n):
                    if p[k] != -1:
                        loc_k = p[k]
                        inc_cost += a[i, k] * b[l, loc_k]
                        inc_cost += a[k, i] * b[loc_k, l]
                inc_cost += a[i, i] * b[l, l]
                
                # Add linear estimate
                # C[i, l] is sum_k a[i, k] b[l, k].
                # This includes k=i and k=assigned and k=unassigned.
                # We already accounted for assigned k in inc_cost (partially, inc_cost has 2 terms, C has 1).
                # Let's just use C[i, l] as a tie-breaker or additive term.
                
                total_est = inc_cost + 0.5 * C[i, l]
                
                if total_est < best_cost:
                    best_cost = total_est
                    best_loc = l
            
            if best_loc == -1:
                best_loc = unassigned_locs[0]
            
            p[i] = best_loc
            unassigned_locs.remove(best_loc)
            
        return p

    # Generate initial solutions
    inits = []
    
    # 1. Linear Greedy
    p_lin = linear_greedy_init(0)
    inits.append(p_lin)
    
    # 2. Random Greedy (shuffle facilities before assigning)
    p_rand_greedy = np.empty(n, dtype=np.int64)
    # Just use random permutation as a baseline
    inits.append(np.random.permutation(n))
    
    # 3. Hungarian (if time permits, though greedy is faster)
    # We'll stick to Greedy + ILS
    
    best_cost = np.inf
    best_p = None
    
    # Refine each init
    for p_init in inits:
        if time.time() > deadline - 20.0:
            break
            
        p_work = p_init.copy()
        # Ensure it's a valid permutation? linear_greedy produces valid permutation.
        cost = compute_cost(p_work)
        
        p_work, cost = local_search_2opt(p_work, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
            
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)
        
    # Iterated Local Search
    rng = np.random.RandomState(42)
    no_improve = 0
    
    while time.time() < deadline - 5.0:
        p = best_p.copy()
        
        # Perturb
        # Number of swaps based on no_improve
        num_swaps = 2 + no_improve // 5
        if num_swaps > 10: num_swaps = 10
        
        for _ in range(num_swaps):
            i = rng.randint(n)
            j = rng.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
                
        cost = compute_cost(p)
        p, cost = local_search_2opt(p, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        # Occasionally do a "large" jump
        if no_improve % 10 == 0:
             # Random restart?
             # p = np.random.permutation(n)
             # cost = compute_cost(p)
             # p, cost = local_search_2opt(p, cost, deadline)
             # if cost < best_cost: ...
             pass

    return best_p.tolist()

# EVOLVE-BLOCK-END
