"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    MOD = 1009
    
    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 = 1 mod 1009 -> x=1 or x=1008
    # These cannot appear in any valid interval
    
    best_start = 2
    best_len = 0
    
    # For each start position s (not self-inverse), find max valid interval length
    for s in range(2, MOD - 1):  # s from 2 to 1007 (skip 1 and 1008)
        if time.time() > deadline:
            break
        
        # Extend interval [s, e] as far as possible
        # For each e, conflict occurs if s <= inv[e] < e (inv[e] is in [s, e-1])
        # or if e is self-inverse (but e >= s >= 2, and 1008 is the only self-inverse >= 2)
        max_len = 0
        for e in range(s, MOD):
            if e == 1008:  # self-inverse, cannot include
                break
            # Check if inv[e] falls in [s, e-1], meaning a prior element in the interval
            # is the inverse of e, creating a forbidden pair
            if inv[e] >= s and inv[e] < e:
                break
            max_len = e - s + 1
        
        if max_len > best_len:
            best_len = max_len
            best_start = s
    
    if best_len == 0:
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
