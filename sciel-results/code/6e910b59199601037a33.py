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
        # First Improvement Local Search with vectorized pair evaluation
        # We evaluate all pairs for a given i, find best improving move, apply it, repeat.
        # This is "Best Improvement" per i, but we restart after each move.
        
        while True:
            if time.time() - start_time > timeout:
                break
            
            moved = False
            for i in range(n):
                if time.time() - start_time > timeout:
                    break
                
                # Compute deltas for swapping i with all j > i
                # We want min delta (most negative)
                u = p[i]
                
                # For each j, delta = 2 * (sum_k (a[i,k]-a[j,k]) * (b[p[j],p[k]] - b[u,p[k]])) - corrections
                # Let's vectorize over j.
                # diff_a_j = a[i] - a[j] for each j. This is (n, n) if we do all j.
                # But we only care about j > i.
                
                # Efficient approach:
                # Let A_i = a[i] (vector of length n)
                # Let B_u = b[u, p] (vector of length n)
                # For a specific j, v = p[j]
                # delta_j = 2 * ( sum_k (a[i,k] - a[j,k]) * (b[v, p[k]] - B_u[k]) ) - 2*(term_i + term_j)
                # = 2 * ( sum_k a[i,k](b[v,p[k]] - B_u[k]) - sum_k a[j,k](b[v,p[k]] - B_u[k]) ) - corrections
                
                # Let X_j = sum_k a[j,k] * (b[p[j], p[k]] - b[u, p[k]])
                # Let Y = sum_k a[i,k] * (b[p[j], p[k]] - b[u, p[k]])
                # Note Y depends on j because p[j] changes.
                
                # This is hard to fully vectorize over j without loops because b[p[j], p[k]] is a 2D lookup.
                # However, we can precompute S_j = sum_k a[j,k] * b[p[j], p[k]]? No, that's not right.
                
                # Let's stick to the previous loop but maybe optimize the inner part?
                # The previous loop was:
                # for j in range(i+1, n):
                #    delta = get_delta(p, i, j)
                
                # get_delta does:
                # b_v_p = b[v, p]
                # b_u_p = b[u, p]  <-- This is constant for fixed i!
                # diff_b = b_v_p - b_u_p
                # diff_a = a[i] - a[j]
                # sum_all = np.sum(diff_a * diff_b)
                
                # So for fixed i, b_u_p is constant.
                # We can precompute b_u_p = b[u, p]
                # Then for each j:
                #   v = p[j]
                #   diff_b = b[v, p] - b_u_p
                #   diff_a = a[i] - a[j]
                #   sum_all = np.dot(diff_a, diff_b)
                #   term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                #   term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                #   delta = 2 * (sum_all - term_i - term_j)
                
                b_u_p = b[u, p]
                a_i = a[i]
                
                best_delta = 0
                best_j = -1
                
                for j in range(i + 1, n):
                    v = p[j]
                    diff_b = b[v, p] - b_u_p
                    diff_a = a_i - a[j]
                    sum_all = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    delta = 2 * (sum_all - term_i - term_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j != -1:
                    # Swap i and best_j
                    j = best_j
                    p[i], p[j] = p[j], p[i]
                    current_cost += best_delta
                    moved = True
                    break # Restart scan from i=0
            
            if not moved:
                break
        
        return p, current_cost

    start_time = time.time()
    timeout = 115.0
    
    # 1. Initial Solution: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Sort facilities by flow (heaviest first)
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    # Sort locations by distance sum (most central first)
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
