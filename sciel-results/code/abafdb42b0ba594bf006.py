import time
import sys
from math import ceil, log2

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
    
    # Precompute popcount
    popcount = [0] * (1 << n)
    for m in range(1, 1 << n):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
            
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
    
    is_valid_part = [False] * (1 << n)
    for m in range(1, 1 << n):
        is_valid_part[m] = is_clique[m] or is_independent[m]
        
    # DP to find minimum number of parts
    # dp[mask] = minimum number of valid parts to partition mask
    # best_sub[mask] = the valid part (submask) chosen in the optimal solution for mask
    INF = 10**9
    dp = [INF] * (1 << n)
    best_sub = [0] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        # Iterate over submasks of mask
        # Optimization: Only consider submasks containing the least significant bit of mask
        # This avoids counting permutations of the same partition
        lsb = mask & -mask
        sub = mask
        while sub:
            if (sub & lsb) != 0 and is_valid_part[sub]:
                # If we pick 'sub' as one part, the rest is 'mask ^ sub'
                # We need dp[mask ^ sub] to be already computed?
                # No, dp is computed for all smaller masks?
                # Actually, mask ^ sub is a proper subset of mask, so its value is already computed
                # if we iterate mask in increasing order?
                # Wait, mask ^ sub < mask is not necessarily true in terms of integer value,
                # but it is a subset. Since we iterate from 1 to 2^n-1,
                # does mask ^ sub always have a smaller integer value?
                # Yes, because sub is non-empty and subset of mask, so mask ^ sub < mask.
                # Proof: Let mask = S. Let sub be a non-empty subset.
                # S ^ sub clears the bits in sub. Thus S ^ sub < S.
                # So dp[S ^ sub] is already computed.
                
                candidate = 1 + dp[mask ^ sub]
                if candidate < dp[mask]:
                    dp[mask] = candidate
                    best_sub[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct the partition
    partition = []
    current_mask = full_mask
    while current_mask:
        sub = best_sub[current_mask]
        # Extract vertices in sub
        verts = []
        m = sub
        while m:
            l = m & -m
            verts.append(l.bit_length() - 1)
            m ^= l
        partition.append(verts)
        current_mask ^= sub
        
    # Convert to list of lists of ints
    result = []
    for part in partition:
        result.append([int(v) for v in part])
        
    return result

def _greedy_construct():
    """Fallback greedy constructor."""
    n = P['n']
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            is_cli = True
            is_ind = True
            for u in part:
                e = (min(u, v), max(u, v))
                if e in edge_set:
                    is_ind = False
                else:
                    is_cli = False
            if is_cli or is_ind:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    
    return parts

# EVOLVE-BLOCK-END
