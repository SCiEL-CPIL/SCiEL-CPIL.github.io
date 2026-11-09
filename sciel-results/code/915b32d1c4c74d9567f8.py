import random

def construct():
    n = 512
    
    # Generate all 4-term arithmetic progressions in [1, n]
    # An AP is defined by (start, d) -> (start, start+d, start+2d, start+3d)
    # We store them as tuples of 0-indexed elements for bitmask compatibility
    all_aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for start in range(1, n - 3 * d + 1):
            a = start - 1
            b = start + d - 1
            c = start + 2 * d - 1
            e = start + 3 * d - 1
            all_aps.append((a, b, c, e))
            
    best_size = 0
    best_mask = 0
    
    # Run multiple iterations with different randomization strategies
    for _ in range(50):
        # Start with all elements included
        mask = (1 << n) - 1
        
        # To speed up, we can maintain a list of "active" AP indices
        # But checking if an AP is active requires checking 4 bits.
        # We can just iterate through all_aps and filter.
        
        # A more efficient way:
        # Keep a list of APs that are currently "bad" (all 4 elements present).
        # Initially, all APs are bad.
        active_aps = list(range(len(all_aps)))
        
        # To avoid O(N_aps) scan every step, we can be smarter, 
        # but N_aps ~ 44000, steps ~ 300. 44000*300 = 13M checks. 
        # Each check is 4 bit ops. This is fine in Python if optimized? 
        # 13M iterations in Python might take ~1-2 seconds. 
        # 50 iterations * 2s = 100s. Too slow.
        
        # We need to optimize the inner loop.
        # Instead of scanning all active_aps, we can pick a random active AP.
        # But we need to update active_aps after removal.
        
        # Alternative: 
        # Use a set of "conflict" APs.
        # When we remove an element x, we remove all APs containing x from the conflict set.
        # We need a mapping: element -> list of AP indices containing it.
        
    # Pre-compute element to AP indices mapping
    elem_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(all_aps):
        for x in ap:
            elem_to_aps[x].append(i)
            
    for _ in range(50):
        mask = (1 << n) - 1
        # active_aps is a set of indices of APs that are fully contained in mask
        active_aps = set(range(len(all_aps)))
        
        while active_aps:
            # Pick a random active AP to resolve
            ap_idx = random.choice(list(active_aps))
            ap = all_aps[ap_idx]
            
            # Find the element in this AP with the highest "degree" 
            # (number of active APs it is part of)
            # Note: The AP itself is active.
            best_x = -1
            max_deg = -1
            
            for x in ap:
                # Count how many active APs contain x
                # We can't just count len(elem_to_aps[x]) because some might not be active.
                # We need to check membership in active_aps.
                deg = 0
                for ap_i in elem_to_aps[x]:
                    if ap_i in active_aps:
                        deg += 1
                
                if deg > max_deg:
                    max_deg = deg
                    best_x = x
            
            # Remove best_x from mask
            mask &= ~(1 << best_x)
            
            # Update active_aps: remove all APs that contain best_x
            # These APs are no longer "active" (bad) because best_x is gone.
            for ap_i in elem_to_aps[best_x]:
                if ap_i in active_aps:
                    active_aps.remove(ap_i)
                    
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
