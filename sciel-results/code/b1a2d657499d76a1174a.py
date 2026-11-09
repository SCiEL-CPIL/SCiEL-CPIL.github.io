def construct():
    # Precompute inverses
    inverses = [0] * 11
    for x in range(1, 11):
        inverses[x] = pow(x, -1, 101)
    
    # min_size[r] stores the minimum number of denominators needed to sum to r
    # best_terms[r] stores the actual terms for that minimum size
    min_size = [11] * 101  # Initialize with a value larger than max possible size (10)
    best_terms = [None] * 101
    
    # Iterate through all non-empty subsets of {1..10}
    for mask in range(1, 1 << 10):
        size = 0
        residue = 0
        terms = []
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                size += 1
                residue = (residue + inverses[x]) % 101
                terms.append(x)
        
        # If this subset size is smaller than the currently known min size for this residue
        if size < min_size[residue]:
            min_size[residue] = size
            best_terms[residue] = terms
    
    # Build result
    result = []
    for r in range(101):
        # All residues must be covered. If not, min_size[r] would be 11.
        # But the problem guarantees a solution exists for k<=10.
        result.append(best_terms[r])
        
    return result
