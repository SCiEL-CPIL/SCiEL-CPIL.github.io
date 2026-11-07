"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    full_mask = (1 << n) - 1
    
    # Precompute valid[mask]: True if mask is clique in G or independent set in G
    valid = [False] * (1 << n)
    valid[0] = True
    
    for mask in range(1, 1 << n):
        verts = [i for i in range(n) if mask & (1 << i)]
        is_c = True
        is_ic = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[j] in adj[verts[i]]:
                    is_ic = False
                else:
                    is_c = False
                if not is_c and not is_ic:
                    break
            if not is_c and not is_ic:
                break
        if is_c or is_ic:
            valid[mask] = True
    
    # Check deadline
    if time.time() > deadline - 5.0:
        # Fallback: greedy
        parts = []
        used = [False] * n
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in set(map(tuple, edges)) for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    # DP
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
            sub = (sub - 1) & mask
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    # If DP didn't complete, fallback
    if dp[full_mask] == INF:
        parts = []
        used = [False] * n
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in set(map(tuple, edges)) for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    # Backtrack
    partition = []
    mask = full_mask
    while mask:
        sub = mask
        found = False
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 == dp[mask]:
                    verts = [i for i in range(n) if sub & (1 << i)]
                    partition.append(verts)
                    mask = prev
                    found = True
                    break
            sub = (sub - 1) & mask
        if not found:
            break
    
    result = [[int(v) for v in part] for part in partition]
    return result

# EVOLVE-BLOCK-END
