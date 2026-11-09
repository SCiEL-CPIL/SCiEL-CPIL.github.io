def _find_greedy_compatible_sets(adj, n, max_sets=20, max_size=5):
    # Build compatibility graph
    # comp[i] is bitmask of vertices compatible with i
    comp = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & adj[j]):
                comp[i] |= (1 << j)
                comp[j] |= (1 << i)
                
    sets_found = []
    
    # Greedy clique finding
    # We'll run this multiple times with different random starts or vertex orderings
    for attempt in range(10):
        current_set = []
        candidates_mask = (1 << n) - 1
        
        while candidates_mask:
            # Pick vertex with max degree in candidates_mask
            # Or random
            if attempt % 2 == 0:
                # Max degree
                max_deg = -1
                v = -1
                temp = candidates_mask
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    deg = (comp[idx] & candidates_mask).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        v = idx
                    temp ^= lsb
            else:
                # Random
                v = (random.getrandbits(n.bit_length()) & candidates_mask).bit_length() - 1
                if v < 0 or v >= n:
                    v = (candidates_mask & -candidates_mask).bit_length() - 1
            
            if v == -1:
                break
                
            current_set.append(v)
            candidates_mask &= comp[v]
            
        if len(current_set) > 1:
            sets_found.append(current_set)
            
        if len(sets_found) >= max_sets:
            break
            
    return sets_found
