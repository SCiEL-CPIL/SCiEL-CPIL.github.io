"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        sum_all = np.dot(diff_a, diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta

    def local_search(p, current_cost):
        # First Improvement Local Search
        # Optimized: Pre-compute rows for a and b where possible
        while True:
            if time.time() - start_time > timeout:
                break
            
            moved = False
            for i in range(n):
                if time.time() - start_time > timeout:
                    return p, current_cost
                
                u = p[i]
                b_u_p = b[u, p]
                a_i = a[i]
                
                best_delta = 0
                best_j = -1
                
                # We only check j > i to avoid duplicate swaps
                for j in range(i + 1, n):
                    v = p[j]
                    # b[v, p] is a row lookup, b_u_p is precomputed
                    diff_b = b[v, p] - b_u_p
                    diff_a = a_i - a[j]
                    
                    # Dot product is faster than element-wise multiply + sum
                    sum_all = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    delta = 2 * (sum_all - term_i - term_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j != -1:
                    p[i], p[best_j] = p[best_j], p[i]
                    current_cost += best_delta
                    moved = True
                    break # Restart scan from i=0 after a move
            
            if not moved:
                break
        
        return p, current_cost

    # 1. Initial Solution: Hungarian Algorithm on Surrogate Cost
    # Surrogate cost: C[i, j] = flow_sums[i] * dist_sums[j]
    # This aligns high-flow facilities with high-distance-sum locations (central in minimization context? 
    # Actually, for minimization, we want high flow * low distance. 
    # Wait, b is distance. High distance sum means "peripheral". 
    # Usually, we want high flow to be close to each other. 
    # The linear surrogate is sum a[i,k] b[p[i], p[k]]. 
    # A common heuristic is to minimize sum flow_i * dist_sum_j.
    # If b is distance, low dist_sum is central. High flow should go to central? 
    # Let's check the standard QAP heuristic. 
    # Often, we sort flows descending and distance sums ascending.
    # The Hungarian method on C[i,j] = flow_sums[i] * dist_sums[j] minimizes the product sum.
    # Since we want to minimize total cost, and total cost correlates with placing high flow pairs close together,
    # placing high flow facilities at locations with low average distance to others (low dist_sum) is a good proxy.
    
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Create surrogate matrix
    # We want to minimize sum_i flow_i * dist_sum_{p[i]}
    # So C[i, j] = flow_sums[i] * dist_sums[j]
    surrogate = np.outer(flow_sums, dist_sums)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(surrogate)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS Loop
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        # Check if we have enough time for a full LS pass (approx 1-2s per pass depending on n)
        if time.time() - start_time > timeout - 2.0:
            break
            
        # Adaptive perturbation
        # If we are stagnating, increase perturbation strength
        if no_improve_count < 5:
            num_swaps = 3
        elif no_improve_count < 15:
            num_swaps = 7
        else:
            num_swaps = 15

        # Perform swaps
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
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
