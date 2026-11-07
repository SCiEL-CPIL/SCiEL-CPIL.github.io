import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    
    # Build adjacency as bitmasks for fast computation
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_distinct = 0
    best_subset = list(range(8))  # Fallback
    
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s safety margin
    
    # Precompute all combinations grouped by size
    # Search from larger k to smaller k (larger subsets more likely to have more distinct degrees)
    # But we must check all k to be safe
    
    # Use bitmask representation for subsets
    # For each k, iterate over all C(n,k) subsets
    
    for k in range(n, 7, -1):  # From 17 down to 8
        for combo in combinations(range(n), k):
            # Check time limit
            if time.time() > deadline:
                break
                
            # Convert to bitmask
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Calculate degrees in induced subgraph using bit operations
            deg_set = set()
            for v in combo:
                # Degree of v in induced subgraph = popcount(adj_mask[v] & mask)
                deg = bin(adj_mask[v] & mask).count('1')
                deg_set.add(deg)
            
            distinct_count = len(deg_set)
            
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_subset = list(combo)
                
                # Early termination: if we found k distinct degrees, that's max for this k
                # But we still need to check other k values
                if distinct_count == k:
                    # Can't do better for this k, but larger k might have more
                    # Continue to next k (which is smaller, so we're going down)
                    pass
            
        if time.time() > deadline:
            break
        
        # If we found a perfect solution (k distinct degrees for k vertices),
        # and k is the max possible, we can stop
        # But max possible distinct degrees is n=17, only achievable if k=17 and all degrees distinct
    
    return best_subset
