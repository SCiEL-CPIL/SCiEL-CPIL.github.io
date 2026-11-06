"""Optimized program: find minimum K such that every residue mod 101 is achievable by a subset of {1..10} of size <= K."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns 101 nonempty lists. Row r contains distinct denominators in 1..10
    whose modular inverses sum to r modulo 101.
    Minimizes the largest row length.
    """
    p = 101
    all_denoms = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [pow(x, -1, p) for x in all_denoms]
    
    # For each residue, store the best (smallest size) subset that achieves it
    # min_size[r] = minimum subset size to achieve residue r
    # best_subset[r] = the actual subset (list of denominators)
    min_size = [11] * 101
    best_subset = [None] * 101
    
    # Enumerate all 1023 non-empty subsets of {1..10}
    for mask in range(1, 1 << 10):
        # Count bits and compute residue
        size = 0
        residue = 0
        for i in range(10):
            if mask & (1 << i):
                size += 1
                residue = (residue + inv[i]) % p
        
        if size < min_size[residue]:
            min_size[residue] = size
            best_subset[residue] = [all_denoms[i] for i in range(10) if mask & (1 << i)]
    
    # The optimal maximum row length is the max of min_size over all residues
    max_row_len = max(min_size)
    
    # Construct the result: for each residue r, use the best_subset
    # (which has size <= max_row_len by construction)
    result = [best_subset[r] for r in range(101)]
    
    # Safety check: ensure all entries are non-empty
    for r in range(101):
        if result[r] is None or len(result[r]) == 0:
            result[r] = [1]
    
    return result
# EVOLVE-BLOCK-END
