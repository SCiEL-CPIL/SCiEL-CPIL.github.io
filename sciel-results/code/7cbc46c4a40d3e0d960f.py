"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency set for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_subset = list(range(min_verts))
    best_count = 0
    
    # Try all subset sizes from largest to smallest
    for k in range(n, min_verts - 1, -1):
        if time.time() > deadline:
            break
        
        target_distinct = k - 1
        
        for subset in combinations(range(n), k):
            if time.time() > deadline:
                break
            
            subset_set = set(subset)
            degrees = []
            for v in subset:
                deg = len(adj[v] & subset_set)
                degrees.append(deg)
            
            distinct_degrees = len(set(degrees))
            
            # If we found the maximum possible distinct degrees for this size, return immediately
            if distinct_degrees == target_distinct:
                return list(subset)
            
            if distinct_degrees > best_count:
                best_count = distinct_degrees
                best_subset = list(subset)
                
    return best_subset
# EVOLVE-BLOCK-END
