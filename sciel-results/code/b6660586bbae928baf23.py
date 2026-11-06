def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_length = 0
    L = 1
    
    for R in range(1, n + 1):
        if inv[R] == R:
            # Self-inverse, cannot be in interval
            # Current valid interval is [L, R-1]
            curr_len = R - L
            if curr_len > best_length:
                best_length = curr_len
                best_start = L
            # Reset L to after R
            L = R + 1
        else:
            inv_R = inv[R]
            # Check if inv_R is in [L, R-1]
            if inv_R < R and inv_R >= L:
                L = inv_R + 1
            
            curr_len = R - L + 1
            if curr_len > best_length:
                best_length = curr_len
                best_start = L
                
    return [best_start, best_length]
