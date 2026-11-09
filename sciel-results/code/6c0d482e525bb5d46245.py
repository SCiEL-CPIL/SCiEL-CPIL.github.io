def construct():
    import random
    
    n = 256
    
    # Build conflict graph as bitmasks
    # adj_mask[v] has bit (u-1) set if v conflicts with u
    adj_mask = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                # Set bits for a and b
                # Bit index for value x is x-1
                adj_mask[a] |= (1 << (b - 1))
                adj_mask[b] |= (1 << (a - 1))
                
    # Initial remaining mask: all bits 0..255 set
    full_mask = (1 << n) - 1
    
    best_set = []
    best_size = 0
    
    # Run randomized greedy multiple times
    # 5000 runs should be fast enough with bitmasks
    for _ in range(5000):
        remaining = full_mask
        current_set = []
        
        while remaining:
            min_deg = n + 1
            candidates = []
            
            # Iterate through all possible nodes
            # We can optimize by only looking at bits set in remaining
            # But N=256 is small, so 1..256 is fine
            for v in range(1, n + 1):
                if remaining & (1 << (v - 1)):
                    deg = (adj_mask[v] & remaining).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
            
            # Pick a random candidate
            v = random.choice(candidates)
            current_set.append(v)
            
            # Remove v and its neighbors from remaining
            # Neighbors of v are in adj_mask[v]
            # We need to remove v itself and all neighbors
            # mask_to_remove = (1 << (v-1)) | adj_mask[v]
            # remaining &= ~mask_to_remove
            mask_to_remove = (1 << (v - 1)) | adj_mask[v]
            remaining &= ~mask_to_remove
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
