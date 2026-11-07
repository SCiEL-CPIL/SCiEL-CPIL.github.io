"""Improved program for the suite task; optimized exhaustive search with bitmasks."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_subset = list(range(min_v))
    best_distinct = 0
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Precompute subset masks for all combinations? No, that's too much memory for 2^17.
    # But we can iterate combinations and compute mask incrementally or just compute it.
    # Computing mask from combination is O(k).
    
    # Optimization: 
    # 1. Iterate k from min_v to n.
    # 2. For each combination, compute mask.
    # 3. Compute degrees using bit_count.
    # 4. Check distinct count.
    # 5. Early termination if best_distinct == n (impossible for k<n, but for k=n it's max n).
    #    Actually max distinct degrees for a subset of size k is k.
    #    So if we find a subset of size k with k distinct degrees, that's optimal for that k.
    #    But a larger k might have more distinct degrees? No, max is size.
    #    So if we find k distinct degrees in a subset of size k, we can't beat it with a larger subset 
    #    unless the larger subset has MORE distinct degrees, which is impossible since max distinct is size.
    #    Wait. If subset size is k, max distinct degrees is k.
    #    If subset size is k+1, max distinct degrees is k+1.
    #    So we should always check larger subsets.
    #    However, if we find 8 distinct degrees in an 8-vertex subset, can we find 9 in a 9-vertex subset? Yes.
    #    So we must search all sizes.
    
    # Can we prune?
    # If best_distinct is D, we only care about subsets that can potentially exceed D.
    # A subset of size k can have at most k distinct degrees.
    # So if k <= best_distinct, we can skip this size entirely?
    # No, because we want to maximize the COUNT of distinct degrees.
    # If best_distinct is 5, and we are looking at k=5, max possible is 5, which is not > 5.
    # So we can skip sizes k where k <= best_distinct.
    
    # Let's implement this pruning.
    
    for k in range(min_v, n + 1):
        if k <= best_distinct:
            continue
        
        if time.time() > deadline:
            break
            
        for subset in itertools.combinations(range(n), k):
            if time.time() > deadline:
                break
                
            # Create a bitmask for this subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
                
            degrees = []
            for v in subset:
                deg = (adj[v] & mask).bit_count()
                degrees.append(deg)
                
            distinct = len(set(degrees))
            
            if distinct > best_distinct:
                best_distinct = distinct
                best_subset = list(subset)
                
                # If we achieved the maximum possible for this size k, 
                # we still need to check larger k because they can have more distinct degrees.
                # But if distinct == n, we are done (global max).
                if best_distinct == n:
                    return [int(x) for x in best_subset]
        if time.time() > deadline:
            break
            
    return [int(x) for x in best_subset]
# EVOLVE-BLOCK-END
