"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_min_vertices = P['minimum_vertices']

# Precompute adjacency as bitmask integers
_adj_mask = [0] * _N
for (u, v) in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

def _precompute_degrees(n, adj_mask):
    """
    Precompute for every subset S (bitmask) the tuple of degrees of vertices in S.
    Uses DP: deg(v, S) = deg(v, S without v) + [v adjacent to any vertex in S without v]
    We store a flat array where entry for (mask, v) is the degree of v in induced subgraph on mask.
    Actually simpler: for each mask, compute a list of degrees for v in mask.
    We'll store degrees as a list of tuples indexed by mask.
    """
    total = 1 << n
    # deg_table[mask] = tuple of degrees for each vertex in mask (in order of vertex index)
    # But that's variable length. Instead, store a single integer per (mask, v) is wasteful.
    # Better: for each mask, compute the multiset of degrees. We only need the set of distinct degrees.
    # Let's precompute: for each mask, the frozenset of degrees. But that's memory heavy.
    # Alternative: precompute for each v and each mask containing v, the degree of v in mask.
    # deg_v_in_mask[v][mask] = popcount(adj_mask[v] & mask)
    # But that's 17 * 131072 = 2.2M entries, each small int. Doable.
    
    # Actually, the fastest approach for the inner loop:
    # For a given mask, we need degrees of all v in mask.
    # deg(v, mask) = popcount(adj_mask[v] & mask)
    # We can precompute a table: for each v, for each mask, popcount(adj_mask[v] & mask)
    # This is 17 * 2^17 = 2,228,224 values. Each is 0..16, fits in a byte.
    # Use numpy uint8 array for speed and memory efficiency.
    import numpy as np
    # Table shape: (n, 1<<n), dtype uint8
    # table[v, mask] = popcount(adj_mask[v] & mask)
    # We can compute this via DP: table[v, mask] = table[v, mask without lsb] + (1 if lsb vertex is adj to v else 0)
    # But simpler: just compute directly with numpy for each v.
    
    # Actually for n=17, the original loop is already fast enough. Let me time it mentally:
    # 131072 masks * ~8.5 avg vertices * bit_count = ~1.1M bit_counts. That's < 0.5s in CPython.
    # The bottleneck is the Python loop overhead, not the bit_count.
    
    # Let's use a different approach: precompute for each mask the sorted list of degrees
    # using vectorized numpy operations on chunks.
    
    pass

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin under 90s kill
    best_score = -1
    best_mask = 0
    
    n = _N
    min_v = _min_vertices
    
    # Precompute popcount for all masks (0 to 2^17 - 1)
    # This avoids calling .bit_count() repeatedly
    total_masks = 1 << n
    popcount = [0] * total_masks
    for i in range(1, total_masks):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute for each vertex v, a lookup: for each mask, popcount(adj_mask[v] & mask)
    # This is the key speedup: O(1) degree lookup instead of O(1) bit_count (same, but avoids the & operation)
    # Actually, let's precompute deg_table[v][mask] = (adj_mask[v] & mask).bit_count()
    # 17 * 131072 = 2.2M entries. Precompute time ~2-3s. Then lookup is O(1).
    
    # Build degree lookup tables
    # deg_lookup[v] is a list of length total_masks, where deg_lookup[v][mask] = popcount(adj_mask[v] & mask)
    deg_lookup = []
    for v in range(n):
        am = _adj_mask[v]
        # For each mask, compute popcount(am & mask)
        # Use the precomputed popcount array: popcount[am & mask]
        # But am & mask can be any value 0..2^17-1, and we have popcount for all of them.
        # So deg_lookup_v[mask] = popcount[am & mask]
        # We can build this as a list comprehension or array.
        # For speed, use a list: [popcount[am & m] for m in range(total_masks)]
        # That's 131072 iterations per vertex, 17 vertices = 2.2M iterations. ~1-2s.
        dl = [0] * total_masks
        for m in range(total_masks):
            dl[m] = popcount[am & m]
        deg_lookup.append(dl)
    
    if time.time() > deadline:
        # Shouldn't happen, but safety
        return list(range(min_v))
    
    # Now enumerate all masks with popcount >= min_v
    # For each, compute distinct degrees using the precomputed lookup
    for mask in range(1, total_masks):
        if time.time() > deadline:
            break
        
        pc = popcount[mask]
        if pc < min_v:
            continue
        
        # Collect distinct degrees
        # Use a small set (degrees are 0..16, so we can use a bitmask of degrees)
        deg_bitmask = 0
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = deg_lookup[v][mask]
            deg_bitmask |= (1 << deg)
            m ^= lsb
        
        score = deg_bitmask.bit_count()
        if score > best_score:
            best_score = score
            best_mask = mask
            if best_score >= n:
                break
                
    # Convert best_mask to list of vertex indices
    result = []
    for v in range(n):
        if best_mask & (1 << v):
            result.append(v)
            
    if not result:
        result = list(range(min_v))
        
    return result
# EVOLVE-BLOCK-END
