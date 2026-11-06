"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    MOD = 1009
    
    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 = 1 mod 1009 -> x=1 or x=1008
    # These cannot appear in any valid interval
    
    best_start = 2
    best_len = 1  # [2,2] is valid: 2*2=4 != 1 mod 1009
    
    # For each start position, find the maximum valid interval length
    for s in range(2, MOD - 1):  # skip 1 and 1008 as starts
        if time.time() > deadline:
            break
        
        # Use a boolean array to track which values are in the current interval
        in_interval = bytearray(MOD)  # all zeros initially
        
        max_len = 0
        for e in range(s, MOD):  # extend interval from s to e
            if time.time() > deadline:
                break
            
            # Check if e is self-inverse (1 or 1008)
            if e == 1 or e == 1008:
                break
            
            # Check if e's inverse is already in the interval
            inv_e = inv[e]
            if in_interval[inv_e]:
                break
            
            # Add e to interval
            in_interval[e] = 1
            max_len = e - s + 1
        
        if max_len > best_len:
            best_len = max_len
            best_start = s
    
    if best_len == 0:
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
