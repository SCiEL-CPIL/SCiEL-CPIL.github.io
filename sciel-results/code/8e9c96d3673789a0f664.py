"""Cochromatic partition via subset DP over precomputed valid parts."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _is_clique_mask(mask):
    """Check if vertices in mask form a clique."""
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        if (adj[v] & m) != m:
            return False
    return True

def _is_independent_mask(mask):
    """Check if vertices in mask form an independent set."""
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        if (adj[v] & m) != 0:
            return False
    return True

def construct():
    n = N
    full_mask = (1 << n) - 1
    
    # Precompute all valid part masks
    valid_parts = []
    for mask in range(1, 1 << n):
        if _is_clique_mask(mask) or _is_independent_mask(mask):
            valid_parts.append(mask)
    
    # Sort by size descending for better pruning
    valid_parts.sort(key=lambda m: bin(m).count('1'), reverse=True)
    
    # DP: dp[mask] = min parts to cover vertices in mask
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    # Store which part was used to reach each state for reconstruction
    choice = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Try each valid part that is a subset of mask
        for part in valid_parts:
            if (part & mask) == part:
                prev = mask ^ part
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    choice[mask] = part
    
    # Reconstruct the partition
    partition = []
    mask = full_mask
    while mask:
        part = choice[mask]
        if part == 0:
            break
        verts = [v for v in range(n) if (part >> v) & 1]
        partition.append(verts)
        mask ^= part
    
    # Convert to list of lists with Python ints
    result = [list(p) for p in partition]
    
    return result
# EVOLVE-BLOCK-END
