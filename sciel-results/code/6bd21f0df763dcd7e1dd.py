import time
import sys
from math import ceil

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    num_masks = 1 << n
    
    # Precompute popcount
    popcount = [0] * num_masks
    for m in range(1, num_masks):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * num_masks
    is_independent = [False] * num_masks
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, num_masks):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
    
    # A mask is a valid part if it's a clique or independent set
    is_valid = [False] * num_masks
    for m in range(1, num_masks):
        is_valid[m] = is_clique[m] or is_independent[m]
    
    # Exact DP: dp[mask] = minimum number of parts to partition the vertex set 'mask'
    # dp[0] = 0
    # dp[mask] = min over valid submasks S of mask: dp[mask ^ S] + 1
    INF = n + 1
    dp = [INF] * num_masks
    dp[0] = 0
    parent = [0] * num_masks  # parent[mask] = the valid submask chosen for mask
    
    # Iterate masks in increasing popcount order
    # For each mask, try all valid submasks
    # Optimization: only iterate submasks that are valid
    for mask in range(1, num_masks):
        # Iterate all submasks of mask
        sub = mask
        best_val = INF
        best_sub = 0
        while sub:
            if is_valid[sub]:
                val = dp[mask ^ sub] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 1:
                        break
            sub = (sub - 1) & mask
        dp[mask] = best_val
        parent[mask] = best_sub
    
    # Reconstruct the partition
    partition = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        # Extract vertices from sub
        verts = []
        m = sub
        while m:
            lsb = m & -m
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        partition.append([int(v) for v in verts])
        mask ^= sub
    
    return partition

# EVOLVE-BLOCK-END
