"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time

def construct():
    t0 = time.time()
    DEADLINE = 88.0 # Safety margin for evaluator 90s limit
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmask
    adj_mask = [0] * n
    for a, b in edges:
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
    
    # Precompute clique/independent set status
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        if is_clique[rest]:
            is_clique[mask] = (adj_mask[v] & rest) == rest
        # else remains False
        
        if is_indep[rest]:
            is_indep[mask] = (adj_mask[v] & rest) == 0
        # else remains False
        
        if time.time() > DEADLINE:
            # Fallback: return a greedy solution if we run out of time (unlikely)
            # But we need to return *something*.
            pass

    valid = [False] * size
    for m in range(1, size):
        valid[m] = is_clique[m] or is_indep[m]
    
    full_mask = size - 1
    
    # DP: dp[mask] = min parts to partition mask
    # Initialize with a large number
    INF = n + 1
    dp = [INF] * size
    dp[0] = 0
    parent = [0] * size
    
    # Iterate masks in increasing order of popcount? 
    # Standard subset DP iterates mask from 1 to 2^n-1.
    # This ensures that when we compute dp[mask], dp[sub] for sub < mask is already computed?
    # No, sub = mask ^ lb... sub is a submask, so sub < mask is NOT guaranteed in integer value, 
    # but sub is a proper submask, so its popcount is smaller.
    # However, dp[rest] where rest = mask ^ sub.
    # rest is a submask of mask.
    # If we iterate mask from 1 to 2^n-1, do we guarantee dp[rest] is computed?
    # rest < mask is not always true (e.g., mask=10 (1010), rest=8 (1000) is < 10. mask=12 (1100), rest=8 (1000) < 12).
    # Actually, for any submask `sub` of `mask`, `sub <= mask` is true?
    # Yes, `sub & mask == sub` implies `sub <= mask`?
    # 1010 (10), sub 1000 (8). 8 < 10.
    # 1100 (12), sub 0100 (4). 4 < 12.
    # It is true that for any proper submask `s` of `m`, `s < m` is NOT necessarily true in integer value?
    # Wait. 1010 (10). Submasks: 0, 2, 8, 10.
    # 8 < 10. 2 < 10.
    # 1001 (9). Submasks: 0, 1, 8, 9.
    # 8 < 9. 1 < 9.
    # Yes, if `s` is a submask of `m`, then `s <= m`.
    # Proof: `m = s | (m & ~s)`. Since `s` and `m & ~s` are disjoint, `m = s + (m & ~s)`.
    # Since `m & ~s >= 0`, `m >= s`.
    # So iterating `mask` from 1 to 2^n-1 ensures that `dp[rest]` (where `rest` is a submask of `mask`) has already been computed.
    
    for mask in range(1, size):
        best_val = INF
        sub = mask
        while sub:
            if valid[sub]:
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    parent[mask] = sub
                    if best_val == 1:
                        break
            sub = (sub - 1) & mask
        
        dp[mask] = best_val
        
        if time.time() > DEADLINE:
            # If we time out, we might not have dp[full_mask]
            # But for n=14, this loop is ~5M iterations, which is < 1s.
            # So this check is just for safety.
            pass

    # Reconstruct
    result_masks = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub == 0:
            # Fallback if DP failed for some reason
            sub = mask
        result_masks.append(sub)
        mask ^= sub
        
    # Convert to lists of vertices
    result = []
    for m in result_masks:
        verts = []
        r = m
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            verts.append(v)
            r ^= lb
        result.append(verts)
        
    return result
# EVOLVE-BLOCK-END
