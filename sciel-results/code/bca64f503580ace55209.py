"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # Generate all non-empty subsets with their residue and size
    # We store (residue, size, subset_list)
    subsets_info = []
    for mask in range(1, 1 << 10):
        terms = []
        residue = 0
        for i in range(10):
            if mask & (1 << i):
                val = i + 1
                terms.append(val)
                residue = (residue + inv[val]) % 101
        subsets_info.append((residue, len(terms), terms))
        
    # Find minimum K such that all residues are covered by subsets of size <= K
    # We sort by size to check coverage incrementally? 
    # Or just iterate K from 1 to 10.
    
    best_K = 10
    best_assignments = {}
    
    for K in range(1, 11):
        covered = {}
        for residue, size, terms in subsets_info:
            if size <= K:
                # If not covered yet, add it. 
                # We just need existence.
                if residue not in covered:
                    covered[residue] = terms
                    
        if len(covered) == 101:
            best_K = K
            best_assignments = covered
            break
            
    # Construct the result
    result = []
    for r in range(101):
        if r in best_assignments:
            result.append(best_assignments[r])
        else:
            # Fallback, should not happen if logic is correct
            result.append([1])
            
    return result
# EVOLVE-BLOCK-END
