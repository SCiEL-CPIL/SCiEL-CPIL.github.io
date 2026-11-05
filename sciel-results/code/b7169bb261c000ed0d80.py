def construct():
    # Direct minimization approach:
    # For each residue, find the subset with the minimum size.
    # This minimizes the maximum row length globally.
    
    min_sizes = [11] * 101  # Initialize with impossible size
    best_sets = [None] * 101
    
    # Precompute inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # Iterate through all non-empty subsets of {1..10}
    for mask in range(1, 1 << 10):
        residue = 0
        size = 0
        terms = []
        
        # Efficiently calculate sum and terms
        # We can unroll or use bit manipulation
        for i in range(10):
            if (mask >> i) & 1:
                denom = i + 1
                residue = (residue + inv[denom]) % 101
                size += 1
                terms.append(denom)
                
        # Update best for this residue
        if size < min_sizes[residue]:
            min_sizes[residue] = size
            best_sets[residue] = terms
            
    # Construct the result
    # If any residue is not covered (size 11), it's a failure, but problem guarantees coverage
    result = []
    for r in range(101):
        if best_sets[r] is None:
            # Fallback, though should not happen
            result.append([1]) 
        else:
            result.append(best_sets[r])
            
    return result
