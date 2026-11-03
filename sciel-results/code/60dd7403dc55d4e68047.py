"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s kill
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    def get_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j in permutation p.
        p[i] -> u, p[j] -> v.
        """
        u = p[i]
        v = p[j]
        
        # Vectorized calculation
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        sum_all = np.dot(diff_a, diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return int(delta)
    
    def local_search_first_improvement(p, current_cost):
        """
        First Improvement Local Search.
        Scans pairs (i, j) in order. If a swap improves cost, apply it and restart scan.
        This is often faster than Best Improvement for large n because it doesn't
        require evaluating all pairs for a given i before moving on.
        """
        while time.time() < timeout:
            improved = False
            for i in range(n):
                if time.time() >= timeout:
                    return p, current_cost
                
                u = p[i]
                b_u_p = b[u, p]
                a_i = a[i]
                
                for j in range(i + 1, n):
                    v = p[j]
                    # Reuse logic from get_delta but inline for speed if possible, 
                    # or just call get_delta. Inlining avoids function call overhead.
                    
                    # b_v_p = b[v, p]
                    # diff_b = b[v, p] - b_u_p
                    # diff_a = a_i - a[j]
                    # sum_all = np.dot(diff_a, diff_b)
                    # term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    # term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    # delta = 2 * (sum_all - term_i - term_j)
                    
                    # Let's just use get_delta for correctness first, then optimize if needed.
                    delta = get_delta(p, i, j)
                    
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        break  # Restart inner loop, which effectively restarts the outer scan logic for this i
                if improved:
                    break # Restart outer loop from i=0
            
            if not improved:
                break
        
        return p, current_cost

    def local_search_best_improvement(p, current_cost):
        """
        Best Improvement Local Search.
        For each i, find the best j > i. If best delta < 0, apply it and restart.
        Slower per pass, but might escape local minima better? Actually, First Improvement
        is usually preferred for speed in ILS. But Best Improvement can be more robust.
        Given the 120s limit, First Improvement allows more iterations.
        """
        while time.time() < timeout:
            improved = False
            best_global_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                if time.time() >= timeout:
                    break
                
                u = p[i]
                b_u_p = b[u, p]
                a_i = a[i]
                
                best_local_delta = 0
                local_best_j = -1
                
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_local_delta:
                        best_local_delta = delta
                        local_best_j = j
                
                if best_local_delta < best_global_delta:
                    best_global_delta = best_local_delta
                    best_i = i
                    best_j = local_best_j
            
            if best_global_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_global_delta
                improved = True
            
            if not improved:
                break
        
        return p, current_cost

    def perturbation(p, current_cost, num_swaps):
        """
        Apply random swaps to perturb the solution.
        """
        for _ in range(num_swaps):
            if time.time() >= timeout:
                break
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = get_delta(p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
        return p, current_cost

    # 1. Initial Solution: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    # Use Best Improvement for the first pass to get a solid local optimum quickly?
    # Or First Improvement?
    # Best Improvement is O(n^2) per pass. First Improvement is O(n^2) worst case too,
    # but often less.
    # Let's use First Improvement as it allows more restarts in ILS.
    p, current_cost = local_search_first_improvement(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS Loop
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    
    # Adaptive perturbation strength
    # Start small, increase if stuck.
    
    while time.time() < timeout:
        # Check time before doing expensive local search
        if time.time() > timeout - 10.0:
            break
            
        # Determine perturbation size
        if no_improve_count < 5:
            num_swaps = 2
        elif no_improve_count < 10:
            num_swaps = 4
        elif no_improve_count < 20:
            num_swaps = 6
        else:
            num_swaps = 8
            
        # Perturb
        p, current_cost = perturbation(p, current_cost, num_swaps)
        
        # Local Search
        # Use First Improvement for speed
        p, current_cost = local_search_first_improvement(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reset to best solution to stay in its neighborhood
            # This is "First Improvement ILS" with reset on failure.
            # Alternative: keep the perturbed solution if it's close?
            # Standard ILS often resets to best to avoid wandering too far.
            p = best_p.copy()
            current_cost = best_cost
            
        # If we are getting close to timeout, do a final polish with Best Improvement
        if time.time() > timeout - 15.0:
             break

    # Final Polish: Use Best Improvement on the best solution found
    # This might squeeze out a bit more if First Improvement got stuck in a shallow local minimum
    # that Best Improvement could navigate? Unlikely, but worth a try if time permits.
    if time.time() < timeout - 2.0:
        p_final = best_p.copy()
        cost_final = best_cost
        p_final, cost_final = local_search_best_improvement(p_final, cost_final)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
