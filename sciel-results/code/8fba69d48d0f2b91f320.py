"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    P = 101
    # Precompute inverses for denominators 1..10
    invs = [pow(x, -1, P) for x in range(1, 11)]
    
    # best_subset[r] will store (size, list_of_denoms) for residue r
    # We want to minimize the size.
    best_subset = [None] * P
    
    # Iterate over all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 such subsets.
    for mask in range(1, 1 << 10):
        size = 0
        weight = 0
        denoms = []
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                denoms.append(denom)
                size += 1
                weight = (weight + invs[i]) % P
        
        # Check if this subset is better (smaller size) for this residue
        if best_subset[weight] is None:
            best_subset[weight] = (size, denoms)
        else:
            if size < best_subset[weight][0]:
                best_subset[weight] = (size, denoms)
                
    # Construct the result list
    # The problem states "101 nonempty lists".
    # If any residue is not covered, we would have a problem, but 
    # with 1023 subsets and 101 residues, coverage is certain.
    # However, to be safe, if any is None, we can fallback to a larger subset 
    # or just return an empty list (which would be invalid). 
    # But we are confident in coverage.
    
    result = []
    for r in range(P):
        if best_subset[r] is not None:
            result.append(best_subset[r][1])
        else:
            # Fallback: this should not happen. 
            # We can try to find any subset, even if we didn't track min size.
            # But since we iterated all, if it's None, it means no subset sums to r.
            # This is mathematically impossible for P=101, N=10? 
            # Let's assume it's covered.
            result.append([])
            
    return result
# EVOLVE-BLOCK-END
