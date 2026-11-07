"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj_masks():
    n = P['n']
    masks = [0] * n
    for u, v in P['edges']:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = P['n']
    min_k = P['minimum_vertices']
    adj_masks = _build_adj_masks()
    
    best_count = 0
    best_subset = None
    
    # We iterate over all subset sizes k from min_k to n
    # Total subsets is small (~66k), so exhaustive search is feasible and optimal.
    for k in range(min_k, n + 1):
        for subset in itertools.combinations(range(n), k):
            if time.time() > deadline:
                return [int(v) for v in best_subset] if best_subset else list(range(min_k))
            
            # Build bitmask for this subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
                
            deg_set = set()
            # Calculate degrees for vertices in the subset
            for v in subset:
                # Degree of v in induced subgraph is number of neighbors of v also in subset
                # This is popcount(adj_mask[v] & mask)
                deg = (adj_masks[v] & mask).bit_count()
                deg_set.add(deg)
                
            cnt = len(deg_set)
            if cnt > best_count:
                best_count = cnt
                best_subset = subset
                
    if best_subset is None:
        return list(range(min_k))
        
    return [int(v) for v in best_subset]

# EVOLVE-BLOCK-END
