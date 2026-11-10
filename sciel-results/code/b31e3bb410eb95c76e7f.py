"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    p = 101
    # Precompute modular inverses for denominators 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # For each residue, find the minimum size of a subset that sums to it
    # and store the mask of that subset
    min_size = [11] * 101
    best_mask = [0] * 101
    
    for mask in range(1, 1 << 10):
        size = 0
        res = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                size += 1
                res = (res + inv[x]) % p
        if size < min_size[res]:
            min_size[res] = size
            best_mask[res] = mask
            
    # Construct the result
    result = []
    for r in range(101):
        mask = best_mask[r]
        terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
        result.append(terms)
        
    return result
# EVOLVE-BLOCK-END
