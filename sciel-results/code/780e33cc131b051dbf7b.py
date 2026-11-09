import random
import time
import numpy as np

def construct():
    def check_validity(n, adj):
        # adj is a list of ints (bitmasks)
        # Returns (is_valid, violating_subset_mask, violating_size)
        
        # We iterate through all masks.
        # Optimization: Only check masks with popcount >= 6.
        # We can generate combinations or iterate all masks.
        # Iterating all masks is simpler.
        
        # Precompute popcount for all masks? 
        # For n=20, 2^20 masks.
        # We can use bin(x).count('1') which is fast in C.
        
        # To speed up, we can check sizes in increasing order?
        # If a size 6 subset is regular, we return immediately.
        # This is likely the most common violation.
        
        # Let's check size 6 first.
        # Number of size 6 subsets in n=20 is C(20,6) = 38760.
        # This is very fast.
        # If no size 6 is regular, check size 7? C(20,7) = 77520.
        # ... up to n.
        
        # Using itertools.combinations is probably faster for small k than iterating all 2^n masks and filtering.
        # Let's use combinations for k=6, 7, ...
        
        import itertools
        
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                # Check regularity
                # degrees[i] = popcount(adj[u] & mask)
                # We can just sum directly for small k?
                # Or use bitmask.
                
                mask = 0
                for u in subset:
                    mask |= (1 << u)
                
                # Calculate degrees
                degs = []
                for u in subset:
                    degs.append(bin(adj[u] & mask).count('1'))
                    
                if len(set(degs)) == 1:
                    return False, mask, k
                    
        return True, 0, 0

    def local_search(n, max_iters=500):
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for _ in range(max_iters):
            valid, v_mask, v_k = check_validity(n, adj)
            if valid:
                return adj
            
            # If not valid, pick an edge in the violating subgraph to flip
            # We need the list of vertices in v_mask
            vertices = [i for i in range(n) if (v_mask >> i) & 1]
            
            # Pick a random edge in the induced subgraph to flip
            # Or pick a random pair
            u, v = random.sample(vertices, 2)
            
            # Flip edge (u, v)
            if (adj[u] >> v) & 1:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return None

    # Try to find a graph for n=20
    start_time = time.time()
    
    # Try local search for n=20
    for attempt in range(10): # 10 restarts
        if time.time() - start_time > 100:
            break
        adj = local_search(20, max_iters=100)
        if adj is not None:
            edges = []
            for u in range(20):
                for v in range(u+1, 20):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': 20, 'edges': edges}

    # Try n=19
    for attempt in range(10):
        if time.time() - start_time > 110:
            break
        adj = local_search(19, max_iters=100)
        if adj is not None:
            edges = []
            for u in range(19):
                for v in range(u+1, 19):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': 19, 'edges': edges}

    # Fallback to smaller n if time permits
    # ...
    
    # Final fallback: return a small valid graph
    # n=5 is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}
