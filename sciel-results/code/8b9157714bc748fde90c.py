"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency as bitmasks for fast bitwise operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    best_count = -1
    best_subset = None
    
    # Precompute deadline
    deadline = time.time() + 50.0  # 50s safety margin out of 60s
    
    # Iterate over all subset sizes from min_vertices to n
    for k in range(min_vertices, n + 1):
        for subset in itertools.combinations(range(n), k):
            # Check time limit periodically (not every iteration for speed)
            if time.time() > deadline:
                if best_subset is not None:
                    return list(best_subset)
                else:
                    return list(range(min_vertices))
            
            # Build bitmask for the subset
            subset_mask = 0
            for v in subset:
                subset_mask |= (1 << v)
            
            # Compute degrees using bit operations
            degrees = []
            for v in subset:
                deg = bin(adj_mask[v] & subset_mask).count('1')
                degrees.append(deg)
            
            distinct_degrees = len(set(degrees))
            
            if distinct_degrees > best_count:
                best_count = distinct_degrees
                best_subset = subset
    
    if best_subset is not None:
        return list(best_subset)
    else:
        return list(range(min_vertices))
# EVOLVE-BLOCK-END
