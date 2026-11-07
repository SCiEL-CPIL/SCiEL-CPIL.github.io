"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Precompute adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute degree in full graph for each vertex
    full_deg = [adj[v].bit_count() for v in range(n)]
    
    # Upper bound on distinct degrees for any subset of size k:
    # Each vertex in the induced subgraph has degree in [0, k-1],
    # so at most k distinct degrees. But we can tighten: the degree
    # of vertex v in subset S is |N(v) ∩ S|, which is between
    # max(0, full_deg[v] - (n - |S|)) and min(full_deg[v], |S|-1).
    # For a simple upper bound, use k (the number of vertices).
    
    best_score = -1
    best_subset = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Enumerate all subsets via bitmask (2^17 = 131072 total)
    # Use Gray code or simple iteration; simple is fine for 131K
    for mask in range(1, 1 << n):
        # Quick size check: skip if fewer than min_vertices
        sz = mask.bit_count()
        if sz < min_vertices:
            continue
        
        # Upper bound: at most sz distinct degrees
        if sz <= best_score:
            continue
        
        # Check deadline periodically (every 4096 iterations)
        if (mask & 0xFFF) == 0 and time.time() > deadline:
            break
        
        # Compute degrees and count distinct
        deg_set = set()
        m = mask
        while m:
            v = (m & (-m)).bit_length() - 1
            m &= m - 1
            deg_set.add((adj[v] & mask).bit_count())
        
        distinct = len(deg_set)
        if distinct > best_score:
            best_score = distinct
            # Extract vertex list
            best_subset = [v for v in range(n) if (mask >> v) & 1]
            
            # If we've hit the theoretical max for this size, we could stop
            # but let's continue to ensure global optimum
            if best_score >= sz:
                # Perfect: all degrees distinct. Can't do better for this size.
                # But a larger size might also achieve its max.
                # Since we're iterating all masks, just continue.
                pass
    
    if best_subset is None:
        return list(range(min_vertices))
    
    return best_subset
# EVOLVE-BLOCK-END
