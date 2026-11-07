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

def _distinct_degrees(subset_mask):
    """Count distinct degrees in the induced subgraph defined by subset_mask."""
    degrees = set()
    for v in range(P['n']):
        if (subset_mask >> v) & 1:
            d = bin(_ADJ[v] & subset_mask).count('1')
            degrees.add(d)
    return len(degrees)

def _bit_count(mask):
    return bin(mask).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n = P['n']
    min_size = P['minimum_vertices']
    
    best_count = 0
    best_mask = (1 << min_size) - 1  # default: first min_size vertices
    
    # Try all subsets of size 8 to 17
    # Use combinations for each size
    for size in range(n, min_size - 1, -1):  # Try larger sizes first (more room for distinct degrees)
        if time.time() > deadline:
            break
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            for v in combo:
                mask |= (1 << v)
            cnt = _distinct_degrees(mask)
            if cnt > best_count:
                best_count = cnt
                best_mask = mask
                # If we've achieved the theoretical max for this size, we can't do better
                if cnt == size:
                    return [v for v in range(n) if (best_mask >> v) & 1]
    
    # Convert best_mask to list of vertices
    result = [v for v in range(n) if (best_mask >> v) & 1]
    return result
# EVOLVE-BLOCK-END
