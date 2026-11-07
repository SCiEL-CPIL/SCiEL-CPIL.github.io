"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency as bitmasks
_ADJ = [0] * P['n']
for u, v in P['edges']:
    _ADJ[u] |= (1 << v)
    _ADJ[v] |= (1 << u)

# Precompute popcount for all masks up to 2^17
_MAX_MASK = 1 << P['n']
_POPCOUNT = [0] * _MAX_MASK
for i in range(1, _MAX_MASK):
    _POPCOUNT[i] = _POPCOUNT[i >> 1] + (i & 1)

def _distinct_degrees(subset_mask):
    """Count distinct degrees in the induced subgraph defined by subset_mask."""
    degrees = set()
    for v in range(P['n']):
        if (subset_mask >> v) & 1:
            # Degree is the number of neighbors of v within the subset
            # _ADJ[v] is the bitmask of all neighbors of v
            # _ADJ[v] & subset_mask is the bitmask of neighbors of v that are in the subset
            # _POPCOUNT of that gives the degree
            d = _POPCOUNT[_ADJ[v] & subset_mask]
            degrees.add(d)
    return len(degrees)

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, use 85s to be safe
    deadline = start_time + 85
    
    n = P['n']
    min_size = P['minimum_vertices']
    
    best_count = 0
    best_mask = (1 << min_size) - 1  # default: first min_size vertices
    
    # Iterate sizes from n down to min_size
    # If we find a subset of size k with d distinct degrees, 
    # any subset of size <= d cannot beat it (max distinct degrees is size).
    # So we can skip sizes k <= best_count.
    
    for size in range(n, min_size - 1, -1):
        if time.time() > deadline:
            break
            
        # Pruning: if size <= best_count, we can't improve
        if size <= best_count:
            continue
            
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
                
            # Construct mask
            mask = 0
            for v in combo:
                mask |= (1 << v)
                
            cnt = _distinct_degrees(mask)
            if cnt > best_count:
                best_count = cnt
                best_mask = mask
                # If we achieved the theoretical maximum for this size, 
                # we can't do better with this size, but larger sizes 
                # were already processed (since we go down). 
                # Wait, we go from n down to min_size.
                # So if we are at size 17 and find 17, we are done.
                # If we are at size 16 and find 16, best_count=16.
                # Next size 15. 15 <= 16, so we skip.
                # This is correct.
                
    # Convert best_mask to list of vertices
    result = [v for v in range(n) if (best_mask >> v) & 1]
    return result

# EVOLVE-BLOCK-END
