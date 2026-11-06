def construct():
    import random
    from math import gcd
    
    N = 512
    
    # Precompute incompatibility masks
    # mask[i] has bit j set if lcm(i, j) > N
    masks = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            l = i * j // gcd(i, j)
            if l > N:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
                
    def greedy_construct(order, masks):
        mask = 0
        for x in order:
            # Check if x is compatible with current mask
            if (masks[x] & mask) == 0:
                mask |= (1 << x)
        return mask

    def local_search_add(mask, masks):
        # Try to add any compatible node
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not (mask & (1 << x)):
                    if (masks[x] & mask) == 0:
                        mask |= (1 << x)
                        improved = True
                        break
        return mask

    def local_search_swap(mask, masks):
        # Try to remove a node and refill
        # To keep it fast, we only try removing nodes that block at least one other node
        # and we limit the refill search
        current_size = bin(mask).count("1")
        best_mask = mask
        best_size = current_size
        
        # Get list of nodes in mask
        nodes_in_mask = [x for x in range(1, N + 1) if (mask & (1 << x))]
        
        for y in nodes_in_mask:
            # Check if removing y helps
            # Quick check: are there nodes x not in mask that are incompatible with y?
            # If not, removing y doesn't unlock any new nodes directly.
            # But it might unlock nodes that were incompatible with y AND some other node?
            # No, if x was incompatible with y, it becomes compatible with mask\{y} only if it was compatible with the rest.
            
            temp_mask = mask & ~(1 << y)
            
            # Greedily add nodes
            # To speed up, we only try adding nodes that were incompatible with y
            # (since others were already compatible with mask, they are compatible with temp_mask)
            # Wait, if x was compatible with mask, it is compatible with temp_mask.
            # So we only need to try x where (masks[x] & mask) != 0 AND (masks[x] & (1<<y)) != 0?
            # Actually, if (masks[x] & mask) != 0, it was blocked.
            # If we remove y, it is unblocked if (masks[x] & (mask ^ (1<<y))) == 0.
            
            # Iterate through all x not in temp_mask
            # This is O(N) per y.
            for x in range(1, N + 1):
                if not (temp_mask & (1 << x)):
                    if (masks[x] & temp_mask) == 0:
                        temp_mask |= (1 << x)
            
            if bin(temp_mask).count("1") > best_size:
                best_size = bin(temp_mask).count("1")
                best_mask = temp_mask
                
        return best_mask

    best_mask = 0
    best_size = 0
    
    # 1. Degree-based greedy
    degrees = [bin(masks[i]).count("1") for i in range(N + 1)]
    order_deg = sorted(range(1, N + 1), key=lambda x: degrees[x])
    m1 = greedy_construct(order_deg, masks)
    m1 = local_search_add(m1, masks)
    m1 = local_search_swap(m1, masks)
    if bin(m1).count("1") > best_size:
        best_size = bin(m1).count("1")
        best_mask = m1

    # 2. Random restarts
    for _ in range(500):
        order = list(range(1, N + 1))
        random.shuffle(order)
        m = greedy_construct(order, masks)
        m = local_search_add(m, masks)
        # Only do swap if size is promising? Or just do it.
        # Swap is O(N^2). 500 * N^2 is fine.
        m = local_search_swap(m, masks)
        if bin(m).count("1") > best_size:
            best_size = bin(m).count("1")
            best_mask = m

    # Convert mask to list
    result = [x for x in range(1, N + 1) if (best_mask & (1 << x))]
    if not result:
        return [1]
    return result
