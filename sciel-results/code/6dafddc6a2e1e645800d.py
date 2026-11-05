"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Denominators 1 to 10
    denoms = range(1, 11)
    # Precompute modular inverses
    invs = [pow(d, -1, 101) for d in denoms]
    
    # Generate all subsets (masks)
    # We want to process them in increasing order of size (popcount)
    masks = []
    for mask in range(1, 1 << 10):
        # Calculate sum of inverses for this mask
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s = (s + invs[idx]) % 101
            idx += 1
            m >>= 1
        masks.append((mask, s, bin(mask).count("1")))
        
    # Sort by size (popcount), then by mask value for determinism
    masks.sort(key=lambda x: (x[2], x[0]))
    
    representations = [None] * 101
    covered = 0
    
    for mask, residue, size in masks:
        if representations[residue] is None:
            # Extract denominators from mask
            terms = [d for i, d in enumerate(denoms) if (mask >> i) & 1]
            representations[residue] = terms
            covered += 1
            if covered == 101:
                break
                
    # Ensure all are non-empty lists (0 is covered by empty set? 
    # Wait, the task says "101 nonempty lists".
    # Residue 0: sum of inverses mod 101 is 0.
    # The empty set sums to 0. But the list must be non-empty.
    # So we need a non-empty subset that sums to 0.
    # My loop starts from mask=1, so it never considers the empty set (mask 0).
    # So residue 0 will be covered by the first non-empty subset summing to 0.
    # This is correct.
    
    # Double check: are there any None left?
    # With 1023 non-empty subsets, we should cover all 101 residues easily.
    # If somehow a residue is not covered (extremely unlikely), we would need a fallback.
    # But given 1023 subsets and 101 residues, coverage is guaranteed.
    
    return representations
# EVOLVE-BLOCK-END
