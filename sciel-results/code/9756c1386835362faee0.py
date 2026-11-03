"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Pre-compute some properties if needed, but a and b are fixed.
    
    def calc_cost(p):
        # p is permutation of 0..n-1
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # We can do this as np.sum(a * b[np.ix_(p, p)])
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j in permutation p.
        p[i] -> u, p[j] -> v.
        """
        u = p[i]
        v = p[j]
        
        # We need: 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # Let diff_a = a[i] - a[j]
        # Let diff_b_k = b[v, p[k]] - b[u, p[k]]
        
        # Vectorized calculation
        # p is array of locations.
        # b_v = b[v, p] gives array of length n: b[v, p[0]], b[v, p[1]], ...
        # b_u = b[u, p]
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        # Total sum over all k
        sum_all = np.sum(diff_a * diff_b)
        
        # Subtract terms for k=i and k=j
        # Term k=i: diff_a[i] * diff_b[i]
        # diff_b[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        
        # Term k=j: diff_a[j] * diff_b[j]
        # diff_b[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta

    def local_search(p, current_cost):
        # First Improvement Local Search - much faster, allows more passes
        improved = True
        while improved:
            improved = False
            
            for i in range(n):
                if time.time() - start_time > timeout:
                    break
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    # Restart scan from beginning after a move
                    break
        
        return p, current_cost

    start_time = time.time()
    timeout = 115.0
    
    # 1. Initial Solution: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Sort facilities by flow (heaviest first)
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    # Sort locations by distance sum (most central first - lowest sum is usually "central" in QAP context? 
    # Actually, we want to minimize sum of products. 
    # High flow should go to low distance? 
    # Let's check correlation. 
    # If a[i,j] is large, we want b[p[i], p[j]] to be small.
    # So heavy facilities (high sum of flows) should be placed at locations that have low sum of distances to others?
    # Yes.
    central_locations = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS Loop with adaptive perturbation
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        # Perturbation: vary the number of swaps based on stagnation
        if no_improve_count < 5:
            num_swaps = 3
        elif no_improve_count < 15:
            num_swaps = 7
        else:
            num_swaps = 12
        
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = get_delta(p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reset p to best_p to stay in the neighborhood of the best solution
            p = best_p.copy()
            current_cost = best_cost
            
        # Check timeout before next iteration
        if time.time() - start_time > timeout - 5.0:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
