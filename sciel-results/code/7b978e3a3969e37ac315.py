def construct():
    p = 1009
    # Precompute modular inverses for 1 to p-1
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    best_len = 0
    best_start = 1
    
    # Iterate over all possible start positions
    for s in range(1, p):
        # Optimization: if the maximum possible length from s is not better than current best, stop
        if (p - 1) - s + 1 <= best_len:
            break
            
        # Try to extend the interval from s
        for e in range(s, p):
            # Check diagonal constraint: x^2 != 1 mod p
            # Solutions to x^2 = 1 mod 1009 are 1 and 1008
            if e == 1 or e == 1008:
                break
                
            # Check off-diagonal constraint:
            # We need to ensure that for the new element e, its inverse inv[e] 
            # is not already in the interval [s, e-1].
            # If inv[e] is in [s, e-1], then e * inv[e] = 1 mod p, which is forbidden.
            if s <= inv[e] < e:
                break
            
            # If we reach here, [s, e] is a valid interval
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s

    return [best_start, best_len]
