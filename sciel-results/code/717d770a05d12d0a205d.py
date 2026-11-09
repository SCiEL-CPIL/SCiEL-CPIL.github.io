"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency as bitmasks for fast intersection
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute popcount table for 17-bit integers
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    best_subset = None
    best_distinct = 0
    
    deadline = time.time() + 85
    
    # Search all subsets of size >= min_v using bitmask iteration
    # This allows us to find the maximum number of distinct degrees
    # We iterate from full mask down to masks with min_v bits set
    
    # Precompute vertex lists for each possible mask size is not needed
    # Instead, we'll use a smarter approach:
    # For each subset size, we need to check all combinations
    
    # Let's use a different strategy:
    # Since n=17, we can enumerate all 2^17 = 131072 subsets
    # For each subset with at least min_v vertices, compute distinct degrees
    
    # But that might be too slow if we do it naively.
    # Let's stick with the combination approach but optimize it.
    
    # Actually, let's try a smarter search:
    # Start with the full graph, then remove vertices one by one
    # to see if we can increase the number of distinct degrees
    
    # Alternative: Use simulated annealing or local search
    # Start with a random subset and try to improve it
    
    # For now, let's use the existing approach but with better pruning
    
    # Precompute degree for each vertex in the full graph
    full_mask = (1 << n) - 1
    full_degrees = [popcount[adj_mask[v] & full_mask] for v in range(n)]
    
    # Try all subsets of size n, n-1, ..., min_v
    # But we can prune: if we've already found size distinct degrees,
    # we only need to check subsets of size > size
    
    # Let's also try a local search approach
    # Start with the full set and try removing vertices
    
    # Actually, let me think about this differently.
    # The maximum possible distinct degrees is n (for the full graph)
    # But we need at least min_v vertices.
    
    # Let's just enumerate all subsets with at least min_v vertices
    # and find the one with maximum distinct degrees
    
    # For efficiency, we'll use the combination approach
    # but with early termination when we find a perfect solution
    
    for size in range(n, min_v - 1, -1):
        # Upper bound: max distinct degrees for size-k subset is k
        if best_distinct >= size:
            break
        
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            for v in combo:
                mask |= (1 << v)
            degrees = set()
            for v in combo:
                degrees.add(popcount[adj_mask[v] & mask])
            ndistinct = len(degrees)
            if ndistinct > best_distinct:
                best_distinct = ndistinct
                best_subset = combo
                if best_distinct == size:
                    break
        if time.time() > deadline:
            break
        if best_distinct == size:
            break
    
    if best_subset is None:
        return list(range(min_v))
    
    return list(best_subset)
# EVOLVE-BLOCK-END
