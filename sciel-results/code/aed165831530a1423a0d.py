import time
import sys

def construct():
    # Graph data from the seed
    P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}
    
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency list as sets for fast intersection
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    best_distinct_count = 0
    best_subset = None
    
    # Precompute bitmask for vertices? Or just iterate combinations?
    # Iterating all masks 0 to 2^17 - 1 is 131072 iterations.
    # For each mask, if popcount >= 8, calculate degrees.
    
    # Helper to count bits
    # Python 3.10+ has int.bit_count(). Assuming modern Python.
    
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    for mask in range(1 << n):
        if time.time() > deadline:
            break
            
        # Count bits in mask
        # Using bin(mask).count("1") is slower but safe. 
        # int.bit_count() is much faster.
        popcount = mask.bit_count()
        
        if popcount < min_vertices:
            continue
            
        # Calculate degrees in induced subgraph
        # A vertex v is in the subset if (mask >> v) & 1
        # Degree of v is count of neighbors u in adj[v] such that (mask >> u) & 1
        
        degrees = []
        for v in range(n):
            if (mask >> v) & 1:
                # Count neighbors in mask
                deg = 0
                for u in adj[v]:
                    if (mask >> u) & 1:
                        deg += 1
                degrees.append(deg)
                
        distinct_degrees = len(set(degrees))
        
        if distinct_degrees > best_distinct_count:
            best_distinct_count = distinct_degrees
            # Extract vertices from mask
            subset = []
            for v in range(n):
                if (mask >> v) & 1:
                    subset.append(v)
            best_subset = subset
            
            # Optimization: If we reach the theoretical max (which is popcount, but distinct degrees <= popcount),
            # and popcount is small, maybe we can't do better? 
            # Max distinct degrees is at most popcount. 
            # If best_distinct_count == popcount, it's perfect for this size.
            # But we want global max. Max possible is 8 (if size 8) or more.
            # Actually, max distinct degrees in a graph of size k is k.
            # So if we find a subset of size 8 with 8 distinct degrees, that's max possible for size 8.
            # But we can have larger subsets. Max distinct degrees for size 17 is 17.
            # However, usually distinct degrees are limited.
            # We just keep searching until deadline.

    if best_subset is None:
        # Fallback to seed if something went wrong
        return list(range(min_vertices))
        
    return best_subset
