<<<<<<< SEARCH
    # Try d values from minimum upward
    # Minimum d for girth g: need d+5 >= g (odd d) or d+4 >= g (even d)
    # For g=10: d=5 (odd, 5+5=10) works
    
    min_d = max(3, g - 5)
    if min_d % 2 == 0:
        min_d += 1  # prefer odd d for better girth guarantee
    
    best_n = float('inf')
    best_edges = None
    
    # Try d = min_d, min_d+2, min_d+4, ... (odd values)
    # Also try even values if they might give smaller graphs
    for d in range(min_d, min_d + 20):  # Try up to 10 values
        if time.time() > deadline - 10:
            break
        
        result = build_luw(d)
        if result is None:
            continue
        
        n, edges = result
        
        # Verify girth
        if not verify_girth(n, edges, g):
            continue
        
        if n < best_n:
            best_n = n
            best_edges = edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: return the minimum d result even if unverified
    result = build_luw(min_d)
    if result is not None:
        return result
    
    # Last resort: return a trivial invalid result
    return 2, [(0, 1)]
=======
    best_n = float('inf')
    best_edges = None
    
    # Try candidates in order of increasing n
    for n_est, q, d in candidates:
        if time.time() > deadline - 10:
            break
        
        gf_add, gf_sub, gf_mul = get_gf_ops(q)
        if gf_add is None:
            continue
            
        result = build_luw(d, q, gf_add, gf_sub, gf_mul)
        if result is None:
            continue
        
        n, edges = result
        
        # Verify girth
        if not verify_girth(n, edges, g):
            continue
        
        if n < best_n:
            best_n = n
            best_edges = edges
            # If we found a graph close to the Moore bound, we can stop early?
            # Moore bound is 161. If we find n < 200, it's excellent.
            if n <= 200:
                break

    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: try a larger d with q=2 if not found
    # q=2, d=9 -> n=1024
    if time.time() < deadline - 20:
        gf_add, gf_sub, gf_mul = get_gf_ops(2)
        result = build_luw(9, 2, gf_add, gf_sub, gf_mul)
        if result is not None:
            n, edges = result
            if verify_girth(n, edges, g):
                return n, edges
                
    # Last resort: return a trivial invalid result
    return 2, [(0, 1)]
>>>>>>> REPLACE
