"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix for O(1) lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute validity of all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    # Precompute adj masks for faster connectivity checks
    adj_mask = [0] * n
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                adj_mask[u] |= (1 << v)
                
    # Fill validity arrays using bit operations
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # v connected to all in rest?
        if (adj_mask[v] & rest) == rest:
            is_clique[mask] = is_clique[rest]
        else:
            is_clique[mask] = False
            
        # v connected to none in rest?
        if (adj_mask[v] & rest) == 0:
            is_independent[mask] = is_independent[rest]
        else:
            is_independent[mask] = False
            
    # Precompute list of valid masks containing each vertex
    valid_masks_for_vertex = [[] for _ in range(n)]
    for mask in range(1, size):
        if is_clique[mask] or is_independent[mask]:
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                valid_masks_for_vertex[v].append(mask)
                m &= m - 1
                
    # DP to find minimum number of parts
    INF = n + 1
    dp = [INF] * size
    choice = [-1] * size
    dp[0] = 0
    
    # Group masks by popcount for correct DP order
    masks_by_popcount = [[] for _ in range(n + 1)]
    for mask in range(size):
        masks_by_popcount[mask.bit_count()].append(mask)
        
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    for k in range(1, n + 1):
        # Check deadline
        if time.time() > deadline:
            break
            
        for mask in masks_by_popcount[k]:
            # Pick a vertex v in mask to anchor
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            
            best_val = INF
            best_sub = -1
            
            # Iterate through all valid masks S containing v
            for S in valid_masks_for_vertex[v]:
                if (S & mask) == S:
                    remaining = mask ^ S
                    val = dp[remaining] + 1
                    if val < best_val:
                        best_val = val
                        best_sub = S
                        if best_val == 1:
                            break
                            
            if best_val < INF:
                dp[mask] = best_val
                choice[mask] = best_sub
                
    # Reconstruct the partition
    partition = []
    current_mask = full_mask
    while current_mask:
        sub = choice[current_mask]
        if sub == -1:
            # Fallback: pick single vertex
            lsb = current_mask & -current_mask
            v = lsb.bit_length() - 1
            sub = 1 << v
            
        # Extract vertices from sub
        part_verts = []
        m = sub
        while m:
            lsb_m = m & -m
            u = lsb_m.bit_length() - 1
            part_verts.append(int(u))
            m &= m - 1
            
        partition.append(part_verts)
        current_mask ^= sub
        
    # Ensure types are strictly Python int and list
    final_partition = []
    for part in partition:
        final_partition.append([int(v) for v in part])
        
    return final_partition

# EVOLVE-BLOCK-END
