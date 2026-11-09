_SUBSETS_6 = []
_SUBSET_EDGE_IDXS = []

def _precompute_subsets(n):
    global _SUBSETS_6, _SUBSET_EDGE_IDXS
    _SUBSETS_6 = []
    _SUBSET_EDGE_IDXS = []
    
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
        
    for subset in itertools.combinations(range(n), 6):
        _SUBSETS_6.append(subset)
        idxs = []
        for k, l, bit in _EDGES_6_BITS:
            u = subset[k]
            v = subset[l]
            idxs.append(edge_idx(u, v))
        _SUBSET_EDGE_IDXS.append(idxs)

def _count_violations_precomputed(colors):
    if len(colors) < 15:
        return 0
    valid_masks = _get_k222_valid_masks()
    violations = 0
    
    # Iterate over precomputed subsets
    # zip(_SUBSET_EDGE_IDXS, _SUBSETS_6) is not needed, just idxs
    
    for idxs in _SUBSET_EDGE_IDXS:
        mask0 = 0
        # Unroll 15 edges
        # idxs[0] to idxs[14]
        # Corresponds to _EDGES_6_BITS[0..14]
        
        # We can use a loop
        for i in range(15):
            if colors[idxs[i]] == 0:
                mask0 |= (1 << i) # Wait, bit in _EDGES_6_BITS is 1<<i?
                # In _EDGES_6_BITS, the bit is 1, 2, 4... which is 1<<0, 1<<1...
                # So yes, mask0 |= (1 << i) works if idxs[i] corresponds to edge i.
                
        if mask0 in valid_masks:
            violations += 1
            
        mask1 = mask0 ^ 0x7FFF
        if mask1 in valid_masks:
            violations += 1
            
    return violations
