def construct():
    p = 1009
    # Precompute inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, p):
        if R == 1 or R == p - 1:
            # The current window [L, R-1] is valid and ends just before the forbidden element R.
            current_len = R - L
            if current_len > best_len:
                best_len = current_len
                best_start = L
            # R is forbidden, so the next potential interval must start after R.
            L = R + 1
        else:
            inv_R = inv[R]
            # If the inverse of R is already in the window [L, R-1], we have a conflict.
            # We must move the start L to be after inv_R to exclude it.
            if inv_R >= L:
                L = inv_R + 1
            # Now the window [L, R] is valid.
            current_len = R - L + 1
            if current_len > best_len:
                best_len = current_len
                best_start = L
                
    return [best_start, best_len]
