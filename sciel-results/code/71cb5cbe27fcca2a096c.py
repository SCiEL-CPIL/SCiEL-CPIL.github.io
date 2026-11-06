"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    # We only care about the minimum size for each residue.
    # Initialize min_size for each residue to infinity (11 is > max possible 10)
    min_size_for_residue = [11] * 101
    best_mask_for_residue = [0] * 101
    
    for mask in range(1, 1 << 10):
        # Calculate size and residue for this mask
        size = 0
        residue = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                size += 1
                residue += inv[x - 1]
        
        residue %= 101
        
        # Update if this is a smaller size for this residue
        if size < min_size_for_residue[residue]:
            min_size_for_residue[residue] = size
            best_mask_for_residue[residue] = mask
            
    # Construct the result
    result = []
    for r in range(101):
        mask = best_mask_for_residue[r]
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        result.append(terms)
        
    return result
# EVOLVE-BLOCK-END
