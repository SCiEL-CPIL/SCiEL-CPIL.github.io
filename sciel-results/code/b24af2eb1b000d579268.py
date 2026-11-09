def _verify_network(network):
    if not network:
        return N <= 1
    
    masks = [m.copy() for m in _get_initial_masks()]
    
    for i, j in network:
        if i >= j:
            return False
        mi = masks[i]
        mj = masks[j]
        # min to i, max to j
        # Using in-place ops to reduce allocations if possible, 
        # but standard assignment is often optimized.
        # Let's stick to standard assignment for clarity and safety, 
        # but ensure we don't alias.
        masks[i] = mi & mj
        masks[j] = mi | mj
        
    for w in range(N - 1):
        if np.any(masks[w] & ~masks[w + 1]):
            return False
            
    return True
