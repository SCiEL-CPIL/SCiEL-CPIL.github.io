"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse check: x*inv[x] % MOD == 1, and if x == inv[x], it's self-inverse
    # For prime MOD, self-inverses are 1 and MOD-1 = 1008
    
    # For each possible start L, find max R such that [L, R] is valid
    # [L, R] is valid iff for all x in [L, R], inv[x] < L or inv[x] > R
    
    best_start = 2
    best_len = 1
    
    for L in range(1, N + 1):
        # If L is self-inverse, skip
        if inv[L] == L:
            continue
        
        # Extend R from L upward
        max_R = L
        # We need to check: for all x in [L, max_R], inv[x] not in [L, max_R]
        # Start with R = L
        valid = True
        # Check x = L: inv[L] must not be in [L, L] i.e., inv[L] != L (already checked)
        # Actually inv[L] != L is guaranteed since we skip self-inverses
        
        # Let's extend R step by step
        for R in range(L, N + 1):
            # Check if adding R makes it invalid
            # We need: for all x in [L, R], inv[x] < L or inv[x] > R
            # It suffices to check: inv[R] not in [L, R] AND for all x in [L, R-1], inv[x] not in [L, R]
            # The second part: if inv[x] was in [L, R-1] it was already caught. 
            # New violation can only come from: inv[R] in [L, R] or inv[x] == R for some x in [L, R-1]
            
            # Check inv[R]:
            if L <= inv[R] <= R:
                # Adding R creates a violation (either self-inverse or pair with existing)
                break
            
            # Check if any x in [L, R-1] has inv[x] == R
            # This means x = inv[R]... no wait. inv[x] == R means x = inv[R] since inv is its own inverse
            # So we need to check if inv[R] is in [L, R-1]
            if L <= inv[R] <= R - 1:
                # This means inv[R] is already in the interval, and inv[inv[R]] = R, so R and inv[R] form a pair
                break
            
            max_R = R
        
        length = max_R - L + 1
        if length > best_len:
            best_len = length
            best_start = L
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
