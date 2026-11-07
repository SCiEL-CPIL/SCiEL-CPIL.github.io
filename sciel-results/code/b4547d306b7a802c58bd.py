import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Use bit_count if available, else bin count
    try:
        (0).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    best_distinct = 0
    best_subset = list(range(8)) # Fallback, should be updated
    
    start_time = time.time()
    deadline = start_time + 85 # Safety margin for 90s kill
    
    # We want to maximize distinct degrees.
    # Max possible distinct degrees for a subset of size k is k.
    # We iterate k from n down to 8.
    # If we find a subset of size k with k distinct degrees, it's the best possible for that size.
    # However, a larger subset might have more distinct degrees?
    # No, max distinct degrees is bounded by the size of the subset.
    # So if we find 17 distinct degrees in a 17-vertex subset, that's the global max.
    # If we find 16 in a 16-vertex subset, that's the max for 16, but 17 is better.
    # So we should check larger k first.
    
    for k in range(n, 7, -1):
        if best_distinct >= k:
            # We can't beat k with a subset of size < k
            # And we already checked larger sizes
            # But wait, if best_distinct is from a larger k, it's already > k.
            # So if best_distinct >= k, we can break?
            # Yes, because for any smaller k', max distinct is k' < k <= best_distinct.
            break
            
        for combo in combinations(range(n), k):
            if time.time() > deadline:
                return best_subset
                
            mask = 0
            for v in combo:
                mask |= (1 << v)
                
            degrees = [popcount(adj_mask[v] & mask) for v in combo]
            distinct_count = len(set(degrees))
            
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_subset = list(combo)
                
            # If we achieved the theoretical maximum for this size k
            if distinct_count == k:
                # This is the best possible for this k.
                # We can break out of the combinations loop for this k?
                # Yes, no other subset of size k can have more than k distinct degrees.
                # But we still need to check smaller k? No, because k is decreasing.
                # If we found k distinct degrees, best_distinct = k.
                # In the next iteration, k' = k-1. Max possible is k-1 < k.
                # So we can break the outer loop as well.
                return best_subset

    return best_subset
