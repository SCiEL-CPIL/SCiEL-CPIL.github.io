import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = P['n']
    # Build adjacency bitmasks
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute valid[mask]: True if mask induces a clique OR an independent set
    valid = [False] * size
    valid[0] = True
    
    for mask in range(1, size):
        is_clique = True
        is_indep = True
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            rest = mask ^ lb
            # Check clique condition: rest must be subset of adj[v]
            if rest & ~adj[v]:
                is_clique = False
            # Check independent set condition: rest must be disjoint from adj[v]
            if rest & adj[v]:
                is_indep = False
            if not is_clique and not is_indep:
                break
            m ^= lb
        if is_clique or is_indep:
            valid[mask] = True
            
    if time.time() > deadline:
        # Fallback if precomputation was too slow (unlikely for n=14)
        return _greedy_fallback()

    # DP over subsets: dp[mask] = min parts to partition mask
    INF = float('inf')
    dp = [INF] * size
    dp[0] = 0
    
    for mask in range(1, size):
        sub = mask
        best = INF
        while sub:
            if valid[sub]:
                prev = dp[mask ^ sub]
                if prev + 1 < best:
                    best = prev + 1
            sub = (sub - 1) & mask
        dp[mask] = best
        
        if time.time() > deadline:
            break

    if dp[full_mask] < INF:
        # Reconstruct partition
        parts = []
        mask = full_mask
        while mask:
            sub = mask
            found = False
            while sub:
                if valid[sub] and dp[mask ^ sub] + 1 == dp[mask]:
                    part = [v for v in range(n) if (sub >> v) & 1]
                    parts.append(part)
                    mask ^= sub
                    found = True
                    break
                sub = (sub - 1) & mask
            if not found:
                # Fallback for safety, though logically unreachable if DP is correct
                parts.append([v for v in range(n) if (mask >> v) & 1])
                break
        return parts
    else:
        return _greedy_fallback()

def _greedy_fallback():
    n = P['n']
    es = set()
    for a, b in P['edges']:
        es.add((a, b))
        es.add((b, a))
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts
