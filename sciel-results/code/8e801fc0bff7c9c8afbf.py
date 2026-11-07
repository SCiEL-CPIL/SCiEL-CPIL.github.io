"""Cochromatic partition via subset DP."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    # Precompute adjacency as bitmask
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if (i, j) in edge_set:
                adj[i] |= (1 << j)
    
    # Enumerate all subsets, check if clique or independent
    # valid_subsets: list of (mask, list_of_vertices)
    valid_subsets = []
    for mask in range(1, 1 << n):
        # Get vertices in mask
        verts = []
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        k = len(verts)
        if k == 1:
            valid_subsets.append((mask, verts))
            continue
        # Check clique: all pairs must be edges
        is_clique = True
        for i in range(k):
            for j in range(i+1, k):
                if (verts[i], verts[j]) not in edge_set:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            valid_subsets.append((mask, verts))
            continue
        # Check independent: no pairs should be edges
        is_ind = True
        for i in range(k):
            for j in range(i+1, k):
                if (verts[i], verts[j]) in edge_set:
                    is_ind = False
                    break
            if not is_ind:
                break
        if is_ind:
            valid_subsets.append((mask, verts))
    
    # DP: dp[mask] = min parts to cover mask
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    # Also store choice for reconstruction
    choice = [None] * (1 << n)
    
    full_mask = (1 << n) - 1
    
    for mask in range(1, 1 << n):
        best = INF
        best_sub = None
        for vmask, verts in valid_subsets:
            if (vmask & mask) == vmask:  # vmask is subset of mask
                prev = dp[mask ^ vmask]
                if prev + 1 < best:
                    best = prev + 1
                    best_sub = vmask
        dp[mask] = best
        choice[mask] = best_sub
    
    # Reconstruct
    parts = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        if sub is None:
            break
        verts = []
        m = sub
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        parts.append(verts)
        mask ^= sub
    
    return parts
# EVOLVE-BLOCK-END
