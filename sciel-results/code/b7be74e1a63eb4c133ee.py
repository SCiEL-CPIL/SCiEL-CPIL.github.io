"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _precompute_best_subset():
    """Precompute the best subset using bitmask DP at module load time."""
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency bitmask for fast lookup
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    total_masks = 1 << n
    best_subset = None
    best_distinct = 0
    
    # Iterate over all masks with at least min_verts bits set
    for mask in range(total_masks):
        # Count bits set
        bit_count = bin(mask).count('1')
        if bit_count < min_verts:
            continue
        
        # Compute distinct degrees in induced subgraph
        degrees = set()
        for v in range(n):
            if mask & (1 << v):
                # Count neighbors of v within the mask
                deg = bin(adj_mask[v] & mask).count('1')
                degrees.add(deg)
        
        distinct_count = len(degrees)
        if distinct_count > best_distinct:
            best_distinct = distinct_count
            best_subset = [v for v in range(n) if mask & (1 << v)]
    
    if best_subset is None:
        best_subset = list(range(min_verts))
    
    return best_subset

# Precompute at module load time
_PRECOMPUTED_BEST = _precompute_best_subset()

def construct():
    return list(_PRECOMPUTED_BEST)
# EVOLVE-BLOCK-END
