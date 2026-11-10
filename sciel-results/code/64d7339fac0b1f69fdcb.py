"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n = P['n']
    edges = set()
    for e in P['edges']:
        edges.add((e[0], e[1]))
        edges.add((e[1], e[0]))
    
    # Precompute adjacency as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) in edges:
                adj[i] |= (1 << j)
    
    # Precompute for each subset whether it's a clique or independent set
    valid = [False] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Check if mask is a clique
        is_clique = True
        is_indep = True
        # Get list of vertices in mask
        verts = []
        for i in range(n):
            if mask & (1 << i):
                verts.append(i)
        
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if (u, v) in edges:
                    is_indep = False
                else:
                    is_clique = False
                if not is_clique and not is_indep:
                    break
            if not is_clique and not is_indep:
                break
        
        if is_clique or is_indep:
            valid[mask] = True
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return each vertex as its own part
        return [[i] for i in range(n)]
    
    # DP: dp[mask] = minimum number of valid parts to cover vertices in mask
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # Also track which subset was used for reconstruction
    parent = [-1] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Try all valid subsets S that are subsets of mask
        sub = mask
        while sub > 0:
            if time.time() > deadline:
                break
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    full_mask = (1 << n) - 1
    min_parts = dp[full_mask]
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask > 0:
        sub = parent[mask]
        if sub == -1:
            # Should not happen, but fallback
            part = [i for i in range(n) if mask & (1 << i)]
            parts.append(part)
            break
        part = []
        for i in range(n):
            if sub & (1 << i):
                part.append(i)
        parts.append(part)
        mask ^= sub
    
    # Convert to list of lists of ints
    result = [list(p) for p in parts]
    return result

# EVOLVE-BLOCK-END
