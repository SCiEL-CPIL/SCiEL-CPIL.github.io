import time

def construct():
    # Precompute modular inverses
    p = 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
        
    # min_len[r] = minimum number of denominators to sum to r
    # We initialize to infinity (or 11)
    min_len = [11] * p
    best_subsets = [None] * p
    
    # Iterate over all non-empty subsets (1 to 1023)
    # For each subset, calculate sum and size
    # If size < min_len[sum], update min_len and best_subsets
    for mask in range(1, 1 << 10):
        # Calculate size and sum
        size = 0
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                size += 1
                s += inv[x]
        s %= p
        
        if size < min_len[s]:
            min_len[s] = size
            # Reconstruct subset
            subset = [x for x in range(1, 11) if mask & (1 << (x - 1))]
            best_subsets[s] = subset
            
    # Check if all covered
    # If some are not covered (should not happen), fill with full set or something?
    # But with 1023 subsets and 101 residues, all should be covered by small sizes.
    
    return [best_subsets[r] for r in range(p)]
