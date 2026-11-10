import sys
import time

def construct():
    # Graph definition from the prompt
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = P['edges']
    
    # Deadline guard (55s out of 60s limit)
    deadline = time.time() + 55.0

    # Build adjacency masks
    # adj_mask[v] is a bitmask where bit u is 1 if (u, v) is an edge
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # non_adj_mask[v] is a bitmask where bit u is 1 if u != v and (u, v) is NOT an edge
    all_mask = (1 << n) - 1
    non_adj_mask = [0] * n
    for v in range(n):
        non_adj_mask[v] = all_mask & ~(1 << v) & ~adj_mask[v]

    # Precompute is_clique and is_independent for all subsets
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    valid = [False] * size
    valid[0] = True
    
    for S in range(1, size):
        # Check time
        if time.time() > deadline:
            # Fallback: return a valid partition (e.g., single vertices)
            # This should never happen for N=14 with this fast DP
            return [[i] for i in range(n)]

        lsb = S & -S
        v = lsb.bit_length() - 1
        R = S ^ lsb
        
        # Check clique
        if is_clique[R]:
            if (adj_mask[v] & R) == R:
                is_clique[S] = True
        
        # Check independent
        if is_independent[R]:
            if (non_adj_mask[v] & R) == R:
                is_independent[S] = True
        
        valid[S] = is_clique[S] or is_independent[S]

    # DP for minimum set cover
    # dp[S] = min parts to cover subset S
    INF = float('inf')
    dp = [INF] * size
    dp[0] = 0
    parent = [-1] * size
    
    for S in range(1, size):
        if time.time() > deadline:
            return [[i] for i in range(n)]

        T = S
        while T > 0:
            if valid[T]:
                remaining = S ^ T
                cost = dp[remaining] + 1
                if cost < dp[S]:
                    dp[S] = cost
                    parent[S] = T
            T = (T - 1) & S
            
    # Reconstruct the partition
    parts = []
    S = size - 1
    while S > 0:
        T = parent[S]
        if T == -1:
            # Fallback
            subset_verts = [i for i in range(n) if (S >> i) & 1]
            parts.append(subset_verts)
            break
        subset_verts = []
        mask = T
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            subset_verts.append(int(v))
            mask ^= lsb
        parts.append(subset_verts)
        S = S ^ T
        
    return parts

# EVOLVE-BLOCK-END
