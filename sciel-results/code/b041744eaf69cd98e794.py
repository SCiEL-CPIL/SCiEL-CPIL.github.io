"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85.0
    
    # Build adjacency as bitmask
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute: for each subset mask, check if it's a clique or independent set
    # A subset is a clique if for every pair (u,v) in the subset, u-v is an edge
    # A subset is independent if for every pair (u,v) in the subset, u-v is NOT an edge
    
    full_mask = (1 << n) - 1
    is_valid = [False] * (1 << n)
    is_valid[0] = True  # empty set is both
    
    for mask in range(1, 1 << n):
        # Check if clique: every vertex in mask must be adjacent to every other vertex in mask
        # Equivalent: for each vertex v in mask, (mask without v) must be subset of adj_mask[v]
        clique = True
        indep = True
        m = mask
        while m:
            lowbit = m & (-m)
            v = lowbit.bit_length() - 1
            m ^= lowbit
            rest = mask ^ lowbit
            if rest & ~adj_mask[v]:
                clique = False
            if rest & adj_mask[v]:
                indep = False
            if not clique and not indep:
                break
        is_valid[mask] = clique or indep
    
    # DP: dp[mask] = minimum number of valid parts to cover exactly vertices in mask
    # dp[0] = 0
    # dp[mask] = min over valid nonempty submask S of mask: dp[mask ^ S] + 1
    
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # Also track which valid submask was used for reconstruction
    parent = [0] * (1 << n)  # parent[mask] = the valid submask S used to cover mask
    
    for mask in range(1, 1 << n):
        # Enumerate all nonempty submasks of mask that are valid
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline or dp[full_mask] >= INF:
        # Fallback: trivial partition
        return [[v] for v in range(n)]
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub == 0:
            # Shouldn't happen, but safety
            parts.append([mask.bit_length() - 1])
            mask ^= (1 << (mask.bit_length() - 1))
        else:
            part_vertices = []
            m = sub
            while m:
                lowbit = m & (-m)
                v = lowbit.bit_length() - 1
                part_vertices.append(v)
                m ^= lowbit
            parts.append(part_vertices)
            mask ^= sub
    
    return [list(part) for part in parts]

# EVOLVE-BLOCK-END
