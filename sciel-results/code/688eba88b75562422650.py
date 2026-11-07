"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

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
    
    # Determine max possible distinct degrees
    # Max degree in the full graph
    max_deg = max((adj[v]).bit_count() for v in range(n))
    # In an induced subgraph of size k, degrees range from 0 to k-1
    # So max distinct degrees is min(k, max_deg + 1) for k vertices
    # But actually it's bounded by k (number of vertices) and by max_deg+1
    # Theoretical upper bound for any subset of size k: min(k, max_deg + 1)
    
    best_score = -1
    best_subset = None
    
    # Start with larger subsets where more distinct degrees are possible
    # For size k, max distinct = min(k, max_deg + 1)
    # We want to find the maximum over all k >= min_vertices
    
    # First, check if we can achieve the theoretical max
    # Theoretical max for the largest subset (n vertices) is min(n, max_deg + 1)
    theoretical_max = min(n, max_deg + 1)
    
    # Enumerate subsets, prioritizing larger sizes first
    # since they have higher theoretical maximums
    for size in range(n, min_vertices - 1, -1):
        # Check if this size can beat current best
        size_max = min(size, max_deg + 1)
        if size_max <= best_score:
            # No subset of this size can beat current best
            # Since we're going from large to small, and size_max is non-decreasing
            # with size, we can break
            break
            
        # Enumerate all subsets of this size
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
                
            degrees = []
            for v in subset:
                deg = (adj[v] & mask).bit_count()
                degrees.append(deg)
                
            distinct_degrees = len(set(degrees))
            
            if distinct_degrees > best_score:
                best_score = distinct_degrees
                best_subset = list(subset)
                
                if best_score >= theoretical_max:
                    # Found the global optimum
                    return best_subset
    
    if best_subset is None:
        return list(range(min_vertices))
        
    return best_subset
# EVOLVE-BLOCK-END
