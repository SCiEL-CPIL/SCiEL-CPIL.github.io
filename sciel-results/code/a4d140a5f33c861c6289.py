"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    
    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 = 1 mod 1009 -> x=1 or x=1008
    # These cannot appear in any valid interval
    
    best_start = 2
    best_len = 0
    
    for s in range(2, MOD - 1):  # start from 2 to 1007 (skip self-inverses 1, 1008)
        # Build interval starting at s
        # Track which values are in the interval to check inverse conflicts
        # Interval is [s, s+L-1], we extend L until conflict or end
        max_len = 0
        in_interval = set()
        for e in range(s, MOD):  # end from s to 1008
            # Check if e is self-inverse
            if e == 1 or e == 1008:
                break
            # Check if e is inverse of any element already in interval
            inv_e = inv[e]
            if inv_e in in_interval:
                break
            in_interval.add(e)
            max_len = e - s + 1
        
        if max_len > best_len:
            best_len = max_len
            best_start = s
    
    if best_len == 0:
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
