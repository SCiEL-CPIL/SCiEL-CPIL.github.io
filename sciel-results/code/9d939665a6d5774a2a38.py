"""Rewritten program using a different approach family: greedy + local search + exact DP refinement."""
import random
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute validity for all masks using efficient recurrence
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    valid = [False] * (1 << n)
    valid[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_indep[mask] = is_indep[rest] and ((adj[v] & rest) == 0)
        valid[mask] = is_clique[mask] or is_indep[mask]
    
    # Exact DP to find optimal partition
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)
    parent_sub = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        if valid[mask]:
            dp[mask] = 1
            parent[mask] = 0
            parent_sub[mask] = mask
            continue
        sub = (mask - 1) & mask
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                val = dp[prev] + 1
                if val < dp[mask]:
                    dp[mask] = val
                    parent[mask] = prev
                    parent_sub[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct optimal partition
    parts = []
    mask = full_mask
    while mask:
        sub = parent_sub[mask]
        part = []
        s = sub
        while s:
            v_bit = s & (-s)
            v = v_bit.bit_length() - 1
            part.append(v)
            s ^= v_bit
        parts.append(part)
        mask = parent[mask]
    
    # Ensure parts are lists of Python ints
    return [[int(v) for v in part] for part in parts]
