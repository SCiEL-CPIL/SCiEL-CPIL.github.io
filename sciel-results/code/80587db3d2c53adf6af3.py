"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin

    # Precompute modular inverses for 1..10
    p = 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
        
    # min_size[r] will store the minimum size of a subset summing to r
    # best_mask[r] will store the mask of that subset
    min_size = [float('inf')] * p
    best_mask = [0] * p
    
    # Iterate over all non-empty subsets of {1, ..., 10}
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << 10):
        # Check deadline just in case, though this loop is tiny
        if time.time() > deadline:
            break
            
        residue = 0
        size = 0
        # Unroll or loop to compute sum and size
        for i in range(1, 11):
            if mask & (1 << (i - 1)):
                residue = (residue + inv[i]) % p
                size += 1
        
        if size < min_size[residue]:
            min_size[residue] = size
            best_mask[residue] = mask

    # Construct the result
    result = []
    for r in range(p):
        mask = best_mask[r]
        terms = []
        for i in range(1, 11):
            if mask & (1 << (i - 1)):
                terms.append(i)
        result.append(terms)
        
    return result
# EVOLVE-BLOCK-END
