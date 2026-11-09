"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_subset = list(range(min_verts))
    best_distinct_count = 0
    
    # Time limit guard
    start_time = time.time()
    timeout = 80.0  # 90s limit, keep 10s margin
    
    # Precompute vertex list
    vertices = list(range(n))
    
    # Search subset sizes from largest to smallest
    # Larger subsets have higher potential for distinct degrees
    for k in range(n, min_verts - 1, -1):
        if time.time() - start_time > timeout:
            break
            
        # For size k, max possible distinct degrees is k
        # If we already found >= k distinct degrees, no subset of size <= k can beat it
        if k <= best_distinct_count:
            break
            
        # Iterate over all combinations of k vertices
        for subset in itertools.combinations(vertices, k):
            if time.time() - start_time > timeout:
                break
                
            # Calculate degrees in the induced subgraph using bitmasks
            mask = 0
            for v in subset:
                mask |= (1 << v)
                
            # Early exit: if we already have enough distinct degrees to beat best, 
            # we can stop counting once we've seen best_distinct_count + 1 distinct values
            seen = set()
            distinct_count = 0
            for v in subset:
                deg = (adj[v] & mask).bit_count()
                if deg not in seen:
                    seen.add(deg)
                    distinct_count += 1
                    # Early termination: if distinct_count already exceeds best and 
                    # we can't possibly get more distinct degrees than remaining vertices
                    remaining = len(subset) - len(seen)  # vertices not yet processed
                    if distinct_count + remaining <= best_distinct_count:
                        break
                    if distinct_count > best_distinct_count and distinct_count == len(subset):
                        break  # All degrees distinct, can't do better for this k
            
            if distinct_count > best_distinct_count:
                best_distinct_count = distinct_count
                best_subset = list(subset)
                
                # If all degrees are distinct for this size, this is optimal for this k
                if distinct_count == k:
                    # Can't do better than k for this size, and larger sizes already checked
                    return best_subset
                    
    return best_subset
# EVOLVE-BLOCK-END
