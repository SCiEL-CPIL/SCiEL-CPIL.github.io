from collections import deque

def _psl2_11_elements():
    identity = tuple(range(12))
    
    S = [0]*12
    for i in range(11):
        S[i] = (i+1) % 11
    S[11] = 11
    
    S_inv = [0]*12
    for i in range(11):
        S_inv[i] = (i-1) % 11
    S_inv[11] = 11
    
    T = [0]*12
    T[0] = 11
    T[11] = 0
    for i in range(1, 11):
        inv_i = pow(i, 9, 11)
        T[i] = (11 - inv_i) % 11
        
    visited = {identity}
    queue = deque([identity])
    all_perms = []
    
    def compose(p1, p2):
        return tuple(p1[p2[i]] for i in range(12))
        
    while queue:
        p = queue.popleft()
        all_perms.append(p)
        
        for gen in [S, S_inv, T]:
            np = compose(tuple(gen), p)
            if np not in visited:
                visited.add(np)
                queue.append(np)
                
    return all_perms

def construct():
    start_time = time.time()
    deadline = 135.0
    
    perms = _psl2_11_elements()
    
    base = KNOWN_BASE_BLOCK
    base_mask = 0
    for i in base:
        base_mask |= (1 << i)
        
    orbit_masks = set()
    for p in perms:
        mask = 0
        for i in range(6):
            mask |= (1 << p[base[i]])
        orbit_masks.add(mask)
        
    if len(orbit_masks) == 132:
        # Convert masks back to lists of 6 ints
        result = []
        for m in orbit_masks:
            block = []
            for i in range(12):
                if m & (1 << i):
                    block.append(i)
            result.append(block)
        return result
    
    # Fallback logic remains same but faster?
    # The fallback is unlikely to trigger if the base block is correct.
    # To be safe, we can try a few other base blocks quickly.
    
    candidates = [
        (0, 1, 3, 4, 5, 9),
        (0, 1, 2, 4, 5, 8),
        (0, 1, 3, 4, 6, 10),
        (0, 1, 2, 3, 5, 8),
        (0, 1, 2, 4, 6, 9),
        (0, 1, 3, 5, 6, 10),
        (0, 2, 3, 4, 5, 9),
        (0, 1, 3, 4, 5, 10),
    ]
    
    for base in candidates:
        base_mask = 0
        for i in base:
            base_mask |= (1 << i)
            
        orbit_masks = set()
        for p in perms:
            mask = 0
            for i in range(6):
                mask |= (1 << p[base[i]])
            orbit_masks.add(mask)
            
        if len(orbit_masks) == 132:
            result = []
            for m in orbit_masks:
                block = []
                for i in range(12):
                    if m & (1 << i):
                        block.append(i)
                result.append(block)
            return result
            
    # If all fail, return the best effort from the first one (even if size != 132)
    # Or use the greedy fallback from the original code?
    # The original code had a greedy fallback. I should keep it to be safe,
    # but the bitmask orbit is so fast that it will likely succeed.
    
    # Re-implement greedy fallback using bitmasks for speed?
    # The greedy fallback in the original code was O(924 * 6) checks roughly.
    # It's fast enough.
    
    FIVE_SUBSETS = list(combinations(range(12), 5))
    FIVE_TO_IDX = {s: i for i, s in enumerate(FIVE_SUBSETS)}
    SIX_SUBSETS = list(combinations(range(12), 6))
    SIX_TO_FIVE_IDX = []
    for b in SIX_SUBSETS:
        idxs = [FIVE_TO_IDX[s] for s in combinations(b, 5)]
        SIX_TO_FIVE_IDX.append(idxs)
        
    covered = bytearray(792)
    current_blocks = []
    
    for idx in range(len(SIX_SUBSETS)):
        idxs = SIX_TO_FIVE_IDX[idx]
        if not any(covered[i] for i in idxs):
            current_blocks.append(list(SIX_SUBSETS[idx]))
            for i in idxs:
                covered[i] = 1
                
    return current_blocks
