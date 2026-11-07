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
    
    # Precompute full degrees
    full_degrees = [bin(adj_mask[v]).count('1') for v in range(n)]
    max_full_deg = max(full_degrees)
    
    best_count = -1
    best_subset = None
    
    # Precompute deadline with safety margin
    deadline = time.time() + 45.0
    
    # Helper: compute distinct degrees for a subset given its mask
    def compute_distinct_degrees(subset, subset_mask):
        deg_set = set()
        for v in subset:
            deg = bin(adj_mask[v] & subset_mask).count('1')
            deg_set.add(deg)
        return len(deg_set)
    
    # Search from largest subsets down - larger subsets can have more distinct degrees
    # For a subset of size k, max possible distinct degrees is min(k, max_possible_degree_in_induced+1)
    # But we can use a simpler bound: at most k distinct degrees, and at most max_full_deg+1
    
    # Strategy: try k from n down to min_vertices
    # For each k, if even the theoretical max for this k can't beat best_count, skip
    for k in range(n, min_vertices - 1, -1):
        # Upper bound: can't have more than k distinct degrees, 
        # and degrees in induced subgraph are at most full degree
        theoretical_max_for_k = min(k, max_full_deg + 1)
        if theoretical_max_for_k <= best_count:
            continue  # Can't improve
        
        # Check time periodically
        count = 0
        for subset in itertools.combinations(range(n), k):
            count += 1
            if count % 1000 == 0 and time.time() > deadline:
                break
            
            # Build bitmask for the subset
            subset_mask = 0
            for v in subset:
                subset_mask |= (1 << v)
            
            # Quick upper bound: the number of distinct degrees is at most k
            # Also, if we already have best_count distinct degrees, we need at least
            # one more new degree to improve
            
            # Compute degrees
            deg_set = set()
            for v in subset:
                deg = bin(adj_mask[v] & subset_mask).count('1')
                deg_set.add(deg)
                # Early exit if we've already seen enough distinct degrees
                if len(deg_set) + (k - subset.index(v) - 1) <= best_count:
                    break
            
            distinct_count = len(deg_set)
            
            if distinct_count > best_count:
                best_count = distinct_count
                best_subset = subset
                # If we hit theoretical max for this k, no need to check other k's
                if best_count >= theoretical_max_for_k:
                    # Check if we can do better with larger k
                    # But we're going from large to small, so we've already checked larger k
                    # Actually we should continue checking if larger k could have more
                    # But since we're going from n down, if we found max for this k,
                    # larger k might still have more
                    pass
    
    # If we haven't found anything, try smaller subsets more thoroughly
    if best_count < 5:
        for k in range(min_vertices, min(n, 12)):
            theoretical_max_for_k = min(k, max_full_deg + 1)
            if theoretical_max_for_k <= best_count:
                continue
            
            count = 0
            for subset in itertools.combinations(range(n), k):
                count += 1
                if count % 500 == 0 and time.time() > deadline:
                    break
                
                subset_mask = 0
                for v in subset:
                    subset_mask |= (1 << v)
                
                deg_set = set()
                for v in subset:
                    deg = bin(adj_mask[v] & subset_mask).count('1')
                    deg_set.add(deg)
                
                distinct_count = len(deg_set)
                
                if distinct_count > best_count:
                    best_count = distinct_count
                    best_subset = subset
    
    if best_subset is not None:
        return list(best_subset)
    else:
        return list(range(min_vertices))
# EVOLVE-BLOCK-END
