"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008  # interval is contained in [1, 1008]
    
    # Precompute modular inverses
    # inv[x] is the y in [1, p-1] such that x*y % p == 1
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Use two-pointer sliding window to find the longest valid interval
    # A window [L, R] is valid if:
    #   1. No element in [L,R] is a self-inverse (1 or 1008)
    #   2. No pair (x, inv[x]) both exist in [L,R]
    
    best_len = 0
    best_start = 1
    
    L = 1
    for R in range(1, n + 1):
        # If R is a self-inverse, it cannot be in any valid interval
        # Setting L = R + 1 ensures R is not in the window, and any previous
        # self-inverses < R have already caused L to move past them.
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is in the current window [L, R-1], we have a conflict
        # Move L past invR to resolve the conflict
        if L <= invR < R:
            L = invR + 1
        
        # Check if the window [L, R] is non-empty
        if R >= L:
            curr_len = R - L + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = L
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
