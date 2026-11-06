"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
        
    # For each residue, we want to find the subset with minimum size.
    # If multiple subsets have the same minimum size, any is fine.
    # We store the mask of the best subset for each residue.
    best_mask_for_residue = [-1] * 101
    best_size_for_residue = [11] * 101 # 11 is larger than max size 10
    
    # Iterate through all non-empty subsets
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        # Compute size and sum
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % 101
        
        size = len(terms)
        
        if size < best_size_for_residue[s]:
            best_size_for_residue[s] = size
            best_mask_for_residue[s] = mask
            
    # Determine the minimum K
    # K is the maximum of the best sizes for all residues.
    max_K = 0
    for r in range(101):
        if best_size_for_residue[r] > max_K:
            max_K = best_size_for_residue[r]
            
    # Construct result
    result = [None] * 101
    for r in range(101):
        mask = best_mask_for_residue[r]
        terms = []
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
        result[r] = terms
        
    return result
# EVOLVE-BLOCK-END
