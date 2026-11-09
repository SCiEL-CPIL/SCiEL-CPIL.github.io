"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # We need to assign each residue 0..100 to a subset of {1..10}
    # such that the sum of inverses equals the residue mod 101.
    # We want to minimize the maximum size of these subsets.
    # Since denominators can recur across rows, the problem decomposes:
    # For each residue r, find the subset of {1..10} with minimum size
    # that sums to r. The answer is the max of these minimum sizes.
    
    # Initialize best subset for each residue to None
    best_terms = [None] * 101
    min_popcount = [float('inf')] * 101
    
    # Iterate over all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 such subsets
    for mask in range(1, 1 << 10):
        # Calculate popcount and residue
        popcount = 0
        residue = 0
        terms = []
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                popcount += 1
                residue = (residue + inv[x]) % 101
                terms.append(x)
        
        # Update if this is a better (smaller) subset for this residue
        if popcount < min_popcount[residue]:
            min_popcount[residue] = popcount
            best_terms[residue] = terms
            
    # Verify all residues are covered
    for r in range(101):
        if best_terms[r] is None:
            # This should not happen as 1023 subsets > 101 residues
            # and they are distributed
            raise ValueError(f"Residue {r} not covered")
            
    return best_terms
# EVOLVE-BLOCK-END
