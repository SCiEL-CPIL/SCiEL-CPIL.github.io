"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    deadline = 80.0
    n = P['n']
    edges = P['edges']
    full = (1 << n) - 1

    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    for i in range(n):
        m = 1 << i
        is_clique[m] = True
        is_ind[m] = True

    for mask in range(1, 1 << n):
        if is_clique[mask] or is_ind[mask]:
            continue
        # Pick lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        # Clique: v adjacent to all in rest AND rest is clique
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
        # Independent: v non-adjacent to all in rest AND rest is independent
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True

    # valid[mask] = is_clique[mask] or is_ind[mask]
    valid = [is_clique[m] or is_ind[m] for m in range(1 << n)]

    # DP: dp[mask] = min parts to cover 'mask', choice[mask] = which submask chosen
    INF = 10**9
    dp = [INF] * (1 << n)
    choice = [0] * (1 << n)
    dp[0] = 0

    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    choice[mask] = sub
            sub = (sub - 1) & mask
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Reconstruct
    parts = []
    mask = full
    while mask:
        sub = choice[mask]
        if sub == 0:
            # Fallback: put remaining vertices as singletons
            vertices = [i for i in range(n) if mask & (1 << i)]
            for v in vertices:
                parts.append([v])
            mask = 0
            break
        vertices = [i for i in range(n) if sub & (1 << i)]
        parts.append(vertices)
        mask ^= sub

    result = [list(part) for part in parts]
    return result
# EVOLVE-BLOCK-END
