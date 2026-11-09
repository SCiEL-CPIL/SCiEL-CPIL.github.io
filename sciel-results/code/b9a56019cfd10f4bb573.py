import time
import sys

def construct():
    # Seed data
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute valid_part[mask]: True if mask is a clique or independent set
    size = 1 << n
    valid_part = [False] * size
    valid_part[0] = True  # Empty set is valid (but we won't use it as a part)
    
    # Compute valid_part for all non-empty masks
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check if mask is a clique: v must be connected to all in rest, and rest must be a clique
        # But we need to compute this properly. Let's use a different approach.
        # A set is a clique iff for every pair, there's an edge.
        # A set is independent iff for every pair, there's no edge.
        
        # Efficient check: iterate through vertices in mask
        is_cli = True
        is_ind = True
        m = mask
        while m:
            lsb2 = m & -m
            v2 = lsb2.bit_length() - 1
            rest2 = mask ^ lsb2
            if (adj[v2] & rest2) != rest2:
                is_cli = False
            if (adj[v2] & rest2) != 0:
                is_ind = False
            if not is_cli and not is_ind:
                break
            m ^= lsb2
        
        if is_cli or is_ind:
            valid_part[mask] = True
    
    # Subset DP: dp[mask] = minimum number of parts to cover mask
    # dp[0] = 0
    # dp[mask] = min over all valid submasks s of mask: dp[mask ^ s] + 1
    
    INF = n + 1
    dp = [INF] * size
    dp[0] = 0
    parent = [-1] * size  # parent[mask] = the submask chosen as the part
    
    # Iterate through masks
    for mask in range(1, size):
        # Try all submasks of mask that are valid parts
        # Optimization: only consider submasks that contain the lowest set bit of mask
        # This avoids duplicate work (each partition is counted once)
        lsb = mask & -mask
        sub = mask
        while sub:
            if sub & lsb:  # Only consider submasks containing the lowest bit
                if valid_part[sub]:
                    prev = mask ^ sub
                    if dp[prev] + 1 < dp[mask]:
                        dp[mask] = dp[prev] + 1
                        parent[mask] = sub
            sub = (sub - 1) & mask
    
    full_mask = (1 << n) - 1
    best_num_parts = dp[full_mask]
    
    # Reconstruct the partition
    parts_masks = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub == -1:
            # Should not happen, but fallback
            break
        parts_masks.append(sub)
        mask ^= sub
    
    # Convert masks to lists of vertices
    if parts_masks:
        best_parts = []
        for pm in parts_masks:
            verts = []
            for v in range(n):
                if pm & (1 << v):
                    verts.append(v)
            best_parts.append(verts)
    else:
        # Fallback: each vertex its own part
        best_parts = [[i] for i in range(n)]
    
    # Verify the solution
    def verify(parts):
        seen = set()
        for part in parts:
            if not part:
                return False
            for v in part:
                if v in seen:
                    return False
                seen.add(v)
            # Check if part is clique or independent
            is_cli = True
            is_ind = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    u, w = part[i], part[j]
                    if adj[u] & (1 << w):
                        is_ind = False
                    else:
                        is_cli = False
                    if not is_cli and not is_ind:
                        return False
        return len(seen) == n
    
    if not verify(best_parts):
        # Fallback to greedy
        es = {tuple(e) for e in edges} | {tuple(reversed(e)) for e in edges}
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
        best_parts = parts
    
    return best_parts
