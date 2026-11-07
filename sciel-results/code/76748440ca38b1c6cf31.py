"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    min_v = P['minimum_vertices']
    edges = P['edges']
    
    # 1. Precompute adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # 2. Precompute popcount for all masks (0 to 2^n - 1)
    max_mask = 1 << n
    popcount = [0] * max_mask
    for i in range(1, max_mask):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    best_count = 0
    best_mask = 0
    
    # 3. Exhaustive search with pruning
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin below 60s
    
    for mask in range(1, max_mask):
        # Time check periodically
        if (mask & 4095) == 0 and time.time() > deadline:
            break
            
        k = popcount[mask]
        if k < min_v:
            continue
        
        # Pruning: max possible distinct degrees for k vertices is k
        if k <= best_count:
            continue
            
        # Calculate distinct degrees using bitmask
        deg_mask = 0
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            deg_mask |= (1 << deg)
            m ^= lb
            
        current_count = deg_mask.bit_count()
        
        if current_count > best_count:
            best_count = current_count
            best_mask = mask
            # If we found the theoretical maximum, stop
            if best_count == n:
                break
                
    # 4. Extract vertices from best_mask
    verts = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        verts.append(int(v))
        m ^= lb
    
    return verts
# EVOLVE-BLOCK-END
