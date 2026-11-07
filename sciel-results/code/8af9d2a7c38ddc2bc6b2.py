"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    es = set()
    for e in P['edges']:
        es.add((e[0], e[1]))
        es.add((e[1], e[0]))
    
    adj_mask = [0] * n
    for u, v in es:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute validity of all subsets
    is_valid = [False] * (1 << n)
    is_valid[0] = True
    
    for mask in range(1, 1 << n):
        # Check clique
        is_c = True
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            others = mask ^ low
            if others & ~adj_mask[v]:
                is_c = False
                break
            m ^= low
        
        if is_c:
            is_valid[mask] = True
        else:
            # Check independent set
            is_i = True
            m = mask
            while m:
                low = m & (-m)
                v = low.bit_length() - 1
                others = mask ^ low
                if others & adj_mask[v]:
                    is_i = False
                    break
                m ^= low
            if is_i:
                is_valid[mask] = True

    # DP for minimum number of parts
    INF = 10**9
    dp = [INF] * (1 << n)
    prev = [-1] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if is_valid[sub]:
                cost = dp[mask ^ sub] + 1
                if cost < dp[mask]:
                    dp[mask] = cost
                    prev[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct partition
    partition = []
    mask = full_mask
    while mask:
        sub = prev[mask]
        if sub == -1:
            break
        part = []
        m = sub
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            part.append(int(v))
            m ^= low
        partition.append(part)
        mask ^= sub
        
    return partition

# EVOLVE-BLOCK-END
