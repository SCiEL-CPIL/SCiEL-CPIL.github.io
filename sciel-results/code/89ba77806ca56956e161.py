"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute neighbor count within any subset using bitmask
    # For a subset represented by bitmask, degree of v = popcount(adj_mask[v] & subset_mask)
    
    best_subset = None
    best_count = 0
    
    # Strategy: Use bitmask-based branch-and-bound
    # We'll do a DFS over vertices, deciding include/exclude
    # But with 17 vertices, 2^17 = 131072 subsets, which is very manageable
    # Let's just enumerate all subsets of size >= 8 using bitmasks
    
    # Actually, let's be smarter. We know the max possible distinct degrees is at most k for a k-vertex subgraph.
    # For k=17, max is 17. For k=8, max is 8.
    # Since we want to maximize distinct degrees, we should check larger k first.
    
    # Let's precompute all subsets as bitmasks for sizes >= 8
    # There are sum(C(17,k) for k=8..17) = C(17,8)+...+C(17,17)
    # C(17,8) = 24310, C(17,9) = 24310, C(17,10) = 19448, C(17,11) = 12376, C(17,12) = 6188, C(17,13) = 2380, C(17,14) = 680, C(17,15) = 136, C(17,16) = 17, C(17,17) = 1
    # Total = 24310+24310+19448+12376+6188+2380+680+136+17+1 = 89846 subsets
    
    # This is very fast to enumerate. Let's just do it all.
    
    for k in range(n, min_verts - 1, -1):
        if time.time() > deadline:
            break
        
        for subset in combinations(range(n), k):
            if time.time() > deadline:
                break
            
            # Build subset bitmask
            subset_mask = 0
            for v in subset:
                subset_mask |= (1 << v)
            
            # Compute degrees
            degrees = []
            for v in subset:
                deg = bin(adj_mask[v] & subset_mask).count('1')
                degrees.append(deg)
            
            distinct_degrees = len(set(degrees))
            
            if distinct_degrees > best_count:
                best_count = distinct_degrees
                best_subset = list(subset)
                
                # Early termination: if we found k distinct degrees in a k-vertex subgraph,
                # that's the maximum possible for size k, and no smaller subgraph can beat it
                if best_count == k:
                    return [int(v) for v in best_subset]
    
    if best_subset is None:
        best_subset = list(range(min_verts))
    
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
