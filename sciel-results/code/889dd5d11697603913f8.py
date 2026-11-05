"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        # Check if R is a self-inverse (1 or 1008)
        if R * R % p == 1:
            # R cannot be in any valid interval.
            # Record best so far (window is [L, R-1])
            if R - 1 >= L:
                curr_len = R - 1 - L + 1
                if curr_len > best_len:
                    best_len = curr_len
                    best_start = L
            # Reset window to start after R
            L = R + 1
            continue
            
        # Check if inv[R] is in the current window [L, R-1]
        # If so, we must move L to exclude inv[R]
        if L <= inv[R] < R:
            L = inv[R] + 1
            
        # Now [L, R] is valid
        curr_len = R - L + 1
        if curr_len > best_len:
            best_len = curr_len
            best_start = L
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
