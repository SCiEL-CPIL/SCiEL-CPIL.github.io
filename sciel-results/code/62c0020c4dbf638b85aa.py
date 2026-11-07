"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for overhead
    
    def get_min_k_for_values(vals):
        """
        Uses Branch and Bound to find the minimum K such that 
        there exists a set of 35 edges with <= K distinct sums and <= K distinct products.
        """
        # Pre-compute sums and products for all 91 edges
        edge_data = []
        for i in range(n):
            for j in range(i + 1, n):
                s = vals[i] + vals[j]
                p = vals[i] * vals[j]
                edge_data.append((i, j, s, p))
        
        # Sort edges by sum, then product to cluster similar values
        # This helps B&B find solutions with fewer distinct sums faster
        edge_data.sort(key=lambda x: (x[2], x[3]))
        
        # Theoretical lower bound for K:
        # We need 35 edges.
        # Start K from a reasonable lower bound.
        # Min distinct sums for any 35 edges?
        # Min distinct products for any 35 edges?
        # Start K from 10 and go up.
        
        def exists_valid_set(K):
            sums_set = set()
            prods_set = set()
            selected_edges = []
            
            def backtrack(idx, count):
                if count == m:
                    return True, list(selected_edges)
                
                if idx >= len(edge_data):
                    return False, None
                
                # Pruning: Not enough edges left to reach m
                if count + (len(edge_data) - idx) < m:
                    return False, None
                
                # Pruning: If current distinct counts exceed K
                if len(sums_set) > K or len(prods_set) > K:
                    return False, None
                
                # Option 1: Skip this edge
                found, res = backtrack(idx + 1, count)
                if found:
                    return True, res
                
                # Option 2: Take this edge
                i, j, s, p = edge_data[idx]
                
                # Check if adding this edge exceeds K
                new_sum_count = len(sums_set) + (0 if s in sums_set else 1)
                new_prod_count = len(prods_set) + (0 if p in prods_set else 1)
                
                if new_sum_count <= K and new_prod_count <= K:
                    sums_set.add(s)
                    prods_set.add(p)
                    selected_edges.append((i, j))
                    
                    found, res = backtrack(idx + 1, count + 1)
                    if found:
                        return True, res
                    
                    # Backtrack
                    sums_set.remove(s)
                    prods_set.remove(p)
                    selected_edges.pop()
                    
                return False, None
            
            found, res = backtrack(0, 0)
            return found, res

        # Iterate K upwards
        # Lower bound: 
        # With 35 edges, can we have K=1? No.
        # Let's start from 5.
        for K in range(5, 92):
            found, edges = exists_valid_set(K)
            if found:
                return K, edges
        return 91, None

    # Generate candidate values
    candidates = []
    
    # 1. Arithmetic Progression
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(10, 10 + n)))
    
    # 2. Powers of 2
    vals_pow2 = [2**i for i in range(n)]
    if max(vals_pow2) <= 10000:
        candidates.append(vals_pow2)
        
    # 3. Random candidates
    random.seed(42)
    for _ in range(20):
        vals = random.sample(range(1, 10001), n)
        candidates.append(vals)
        
    # 4. Structured random: Clusters
    for _ in range(20):
        base = random.randint(1, 100)
        step = random.randint(1, 10)
        vals = [base + i * step + random.randint(-2, 2) for i in range(n)]
        if len(set(vals)) == n and min(vals) >= 1 and max(vals) <= 10000:
            candidates.append(vals)

    best_K = float('inf')
    best_edges = None
    best_vals = None

    # Evaluate candidates
    for vals in candidates:
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > 10000:
            continue
            
        K, edges = get_min_k_for_values(vals)
        if K < best_K:
            best_K = K
            best_edges = edges
            best_vals = vals
            
        # Time check
        if time.time() - start_time > time_limit * 0.5:
            break

    # If time remains, do local search on values
    if best_vals and time.time() - start_time < time_limit * 0.8:
        current_vals = list(best_vals)
        current_K = best_K
        
        # Limit iterations to avoid timeout
        max_iters = 50
        for _ in range(max_iters):
            if time.time() - start_time > time_limit * 0.9:
                break
                
            idx = random.randint(0, n - 1)
            old_val = current_vals[idx]
            
            # Perturb
            r = random.random()
            if r < 0.5:
                new_val = old_val + random.randint(-10, 10)
            else:
                new_val = random.randint(1, 10000)
                
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_vals and current_vals.index(new_val) != idx:
                continue
                
            current_vals[idx] = new_val
            
            K, edges = get_min_k_for_values(current_vals)
            
            if K < current_K:
                current_K = K
                best_K = K
                best_edges = edges
                best_vals = list(current_vals)
            else:
                # Revert if not better (Hill Climbing)
                current_vals[idx] = old_val

    if best_vals is None:
        best_vals = list(range(1, n + 1))
        best_K, best_edges = get_min_k_for_values(best_vals)

    return {'values': best_vals, 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
