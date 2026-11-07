"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency as bitmask for fast lookup
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute induced degrees using iterative DP:
    # deg[v][mask] = number of neighbors of v in mask
    # Use the recurrence: deg[v][mask] = deg[v][mask without lsb] + (1 if lsb bit is neighbor of v)
    total_masks = 1 << n
    # For each vertex, store a list of induced degrees for all masks
    # Use bytearray since max degree <= 16 < 256
    induced_deg = [bytearray(total_masks) for _ in range(n)]
    
    for v in range(n):
        am = adj_mask[v]
        arr = induced_deg[v]
        # arr[0] = 0 already
        for mask in range(1, total_masks):
            # Get lowest set bit
            lsb = mask & (-mask)
            prev = mask ^ lsb
            # Check if the bit corresponding to lsb is in adj_mask[v]
            bit_idx = lsb.bit_length() - 1
            if (am >> bit_idx) & 1:
                arr[mask] = arr[prev] + 1
            else:
                arr[mask] = arr[prev]
    
    best_subset = list(range(min_v))
    best_distinct = -1
    
    # Search from largest subsets to smallest (more vertices can yield more distinct degrees)
    for size in range(n, min_v - 1, -1):
        # Prune: max possible distinct degrees is 'size'
        if size <= best_distinct:
            continue
        
        # If best_distinct already equals size, no point checking smaller sizes
        # (but we're going downward, so this won't trigger in this loop direction)
        
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                break
            
            # Compute mask
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Compute degrees using precomputed table
            # Use a small list to avoid set overhead for small sizes
            if size <= 16:
                degs = []
                for v in subset:
                    degs.append(induced_deg[v][mask])
                distinct = len(set(degs))
            else:
                degrees = set()
                for v in subset:
                    degrees.add(induced_deg[v][mask])
                distinct = len(degrees)
            
            if distinct > best_distinct:
                best_distinct = distinct
                best_subset = list(subset)
                # Early exit: can't do better than 'size' distinct degrees for this size
                if best_distinct == size:
                    return best_subset
        
        if time.time() > deadline:
            break
    
    return list(best_subset)

# EVOLVE-BLOCK-END
