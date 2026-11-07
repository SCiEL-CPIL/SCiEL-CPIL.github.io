import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute valid masks
    valid = [False] * (1 << n)
    valid[0] = True
    
    for mask in range(1, 1 << n):
        # Check clique
        is_c = True
        m = mask
        while m and is_c:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if (adj[i] & mask) != (mask ^ lsb):
                is_c = False
            m ^= lsb
        
        if is_c:
            valid[mask] = True
            continue
        
        # Check independent set
        is_i = True
        m = mask
        while m and is_i:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if (adj[i] & mask) != 0:
                is_i = False
            m ^= lsb
        
        if is_i:
            valid[mask] = True

    # DP
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)
    
    # Iterate masks in increasing order of popcount? 
    # Actually, standard subset DP iterates mask from 1 to 2^n-1.
    # dp[mask] depends on dp[mask ^ sub]. 
    # mask ^ sub is a subset of mask, so it has a smaller value (numerically? no, but smaller popcount).
    # If we iterate mask from 1 to 2^n-1, is dp[mask^sub] already computed?
    # mask^sub is a proper subset of mask.
    # Numerically, subsets are not always smaller than the superset.
    # Example: mask=3 (11), sub=2 (10). mask^sub = 1 (01). 1 < 3.
    # Example: mask=5 (101), sub=4 (100). mask^sub = 1 (001). 1 < 5.
    # Generally, if sub is a non-empty subset of mask, mask^sub < mask is NOT always true?
    # Wait. mask = 1010 (10). sub = 0010 (2). mask^sub = 1000 (8). 8 < 10.
    # mask = 1000 (8). sub = 1000 (8). mask^sub = 0.
    # It turns out that for any non-empty proper subset sub of mask, mask ^ sub < mask is NOT guaranteed?
    # Actually, yes, it is.
    # Let mask be a binary number. Removing a set of bits (sub) results in a number with 0s where mask had 1s, and same elsewhere.
    # So the resulting number is strictly smaller than mask.
    # So iterating mask from 1 to 2^n - 1 ensures all subproblems are solved.
    
    for mask in range(1, 1 << n):
        best_val = INF
        best_sub = -1
        
        # Iterate submasks of mask
        sub = mask
        while sub:
            if valid[sub]:
                val = dp[mask ^ sub] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        
        dp[mask] = best_val
        parent[mask] = best_sub

    # Reconstruct
    partition = []
    mask = full_mask
    while mask > 0:
        sub = parent[mask]
        if sub == -1:
            # Fallback: shouldn't happen if valid[full_mask] is not required to be single part
            # But if full_mask itself is valid, parent[full_mask] would be full_mask.
            # If not, it picks a sub.
            break
        part = [i for i in range(n) if sub & (1 << i)]
        partition.append(part)
        mask ^= sub
        
    result = []
    for part in partition:
        result.append([int(v) for v in part])
    
    return result
