"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix for O(1) lookup using bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute validity of all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    # Fill validity arrays
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # v is connected to all in rest if adj_mask[v] & rest == rest
        is_clique[mask] = is_clique[rest] and ((adj_mask[v] & rest) == rest)
        
        # v is connected to none in rest if (adj_mask[v] & rest) == 0
        is_independent[mask] = is_independent[rest] and ((adj_mask[v] & rest) == 0)
        
    # Precompute list of valid masks containing each vertex to speed up DP
    # valid_masks_for_vertex[v] = list of masks S such that v in S and (is_clique[S] or is_independent[S])
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
    # dp[mask] = min parts to cover mask
    INF = n + 1
    dp = [INF] * size
    choice = [-1] * size # choice[mask] = the submask chosen
    dp[0] = 0
    
    # Iterate masks in increasing order of popcount
    masks_by_popcount = defaultdict(list)
    for mask in range(size):
        masks_by_popcount[mask.bit_count()].append(mask)
        
    for k in range(1, n + 1):
        for mask in masks_by_popcount[k]:
            # Pick a vertex v in mask to anchor the submask selection
            # This reduces the number of submasks we check
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            
            best_val = INF
            best_sub = -1
            
            # Iterate through all valid masks S containing v
            for S in valid_masks_for_vertex[v]:
                if (S & mask) == S: # S is subset of mask
                    remaining = mask ^ S
                    if dp[remaining] + 1 < best_val:
                        best_val = dp[remaining] + 1
                        best_sub = S
                        
            if best_val < INF:
                dp[mask] = best_val
                choice[mask] = best_sub
                
    # Reconstruct the partition
    partition = []
    current_mask = full_mask
    while current_mask:
        sub = choice[current_mask]
        if sub == -1:
            # Fallback if something went wrong (should not happen for valid graph)
            # Pick any vertex
            lsb = current_mask & -current_mask
            v = lsb.bit_length() - 1
            sub = 1 << v
            # Ensure sub is valid (single vertex is always clique and independent)
            
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
