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
        # Self-inverses 1 and 1008 cannot be in any valid interval
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # Check if invR is in the current window [L, R-1]
        # If so, we have a conflict. We must move L past invR.
        if L <= invR < R:
            L = invR + 1
        
        # Now check if the window [L, R] is non-empty and valid
        # It is valid by construction:
        # 1. R is not a self-inverse (handled above)
        # 2. invR is not in [L, R-1] (handled above)
        # 3. All pairs in [L, R-1] were already valid (inductive hypothesis)
        # 4. For any x in [L, R-1], if x*y % p == 1 then y = inv[x].
        #    We need to ensure inv[x] is not in [L, R]. Since inv[x] < R (as x >= 2 and 
        #    inv[x] >= 2 for x in [2,1007]), and inv[x] was not in [L, x-1] by the
        #    invariant, we need inv[x] not in [x, R]. But inv[x] != x (no self-inverse
        #    in range), and if inv[x] > x, then when we added inv[x] to the window,
        #    we would have checked if x was in the window and moved L past x if so.
        #    Actually, let me reconsider...
        
        # The key insight: when we add R, we only need to check if inv[R] is in [L, R-1].
        # If it is, we move L past inv[R]. This ensures no pair (x, inv[x]) with both in [L, R].
        # Why? Because for any x in [L, R-1], if inv[x] is in [L, R], then either:
        #   - inv[x] < R: then inv[x] was in [L, R-1] when x was added, so L was moved past inv[x],
        #     meaning inv[x] < L, contradiction.
        #   - inv[x] = R: then x = inv[R], and we just checked/moved L past inv[R] = x,
        #     meaning x < L, contradiction.
        
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
