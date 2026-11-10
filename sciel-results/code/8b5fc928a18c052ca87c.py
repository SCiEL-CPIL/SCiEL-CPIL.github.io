"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    p = 1009
    
    # Precompute modular inverses
    # inv[x] is the unique y in [1, p-1] such that x*y % p == 1
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    # Identify self-inverse elements
    # x is self-inverse if x*x % p == 1
    # For prime p, solutions to x^2 = 1 are x=1 and x=p-1
    self_inv = {1, p - 1}
    
    best_start = 2
    best_len = 0
    
    # Iterate over all possible start positions
    for s in range(1, p):
        if time.time() > deadline:
            break
            
        # If s is self-inverse, it cannot be part of any valid interval
        if s in self_inv:
            continue
            
        # Start with interval [s, s]
        # Since s is not self-inverse, inv[s] != s, so [s, s] is valid.
        e = s
        
        # Extend e as far as possible
        while e < p - 1:  # e can go up to 1007
            next_e = e + 1
            
            # Check if next_e is self-inverse
            if next_e in self_inv:
                break
                
            # Check if adding next_e creates a conflict
            # A conflict exists if inv[next_e] is in [s, next_e]
            # Since next_e is not self-inverse, inv[next_e] != next_e
            # So we just need to check if inv[next_e] is in [s, e]
            inv_next = inv[next_e]
            if s <= inv_next <= e:
                break
            else:
                e = next_e
                
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
