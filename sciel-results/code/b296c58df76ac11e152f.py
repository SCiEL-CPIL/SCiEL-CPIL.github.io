"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inverses = {}
    for x in range(1, 11):
        inverses[x] = pow(x, -1, 101)
        
    # min_size[r] will store the minimum size of a subset that sums to r mod 101
    # best_terms[r] will store the actual list of denominators for that subset
    min_size = [float('inf')] * 101
    best_terms = [None] * 101
    
    # Iterate over all non-empty subsets of {1, ..., 10}
    # There are 2^10 - 1 = 1023 such subsets
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s += inverses[x]
        
        residue = s % 101
        size = len(terms)
        
        # If this subset is smaller than the previously known best for this residue, update
        if size < min_size[residue]:
            min_size[residue] = size
            best_terms[residue] = terms
            
    # Construct the final list of 101 rows
    # Each row r is the list of denominators for the best subset found for residue r
    result = []
    for r in range(101):
        if best_terms[r] is None:
            # This should not happen as the problem guarantees a solution exists
            # and with 1023 subsets covering 101 residues, coverage is dense.
            # However, just in case, we can fallback to any valid subset if needed.
            # But given the math, min_size should be set for all r.
            pass
        result.append(best_terms[r])
        
    return result
# EVOLVE-BLOCK-END
