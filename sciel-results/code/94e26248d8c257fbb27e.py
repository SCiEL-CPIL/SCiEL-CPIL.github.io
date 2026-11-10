"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    
    # Precompute modular inverses using Fermat's little theorem (p is prime)
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Two-pointer approach: for each start s, find max valid end e.
    # Key insight: as s increases, the max valid e is non-decreasing.
    # This is because removing the leftmost element can only reduce conflicts.
    
    best_len = 0
    best_start = 1
    e_max = 0  # Current max valid end for the current s
    
    for s in range(1, p):
        # Ensure e_max is at least s-1 (minimum valid: empty or just s)
        if e_max < s - 1:
            e_max = s - 1
        
        # Try to extend e_max as far as possible for this s
        while e_max + 1 < p:
            next_e = e_max + 1
            # Adding next_e creates a conflict iff inv[next_e] is in [s, next_e]
            if inv[next_e] >= s and inv[next_e] <= next_e:
                break
            e_max = next_e
        
        length = e_max - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]

# EVOLVE-BLOCK-END
