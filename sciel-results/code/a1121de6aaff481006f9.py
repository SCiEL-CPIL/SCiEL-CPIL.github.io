"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute inverses for 1..10 modulo 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # We want to find the minimum size subset for each residue
    # Initialize with a size larger than any possible subset (10)
    min_size = [11] * 101
    best_subset = [None] * 101
    
    # Iterate through all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << 10):
        terms = []
        residue = 0
        size = 0
        for i in range(10):
            if mask & (1 << i):
                val = i + 1
                terms.append(val)
                residue = (residue + inv[val]) % 101
                size += 1
        
        # If this subset is smaller than the current best for this residue, update
        if size < min_size[residue]:
            min_size[residue] = size
            best_subset[residue] = terms
            
    # Construct the result
    result = []
    
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback for safety, though theoretically unreachable
            result.append([1])
            
    return result
# EVOLVE-BLOCK-END
