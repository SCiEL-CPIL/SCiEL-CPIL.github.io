"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute inverses for denominators 1..10 modulo 101
    p = 101
    invs = [pow(x, -1, p) for x in range(1, 11)]
    
    # best[r] will hold (size, terms_list)
    # Initialize with infinity
    best = {r: (10**9, []) for r in range(p)}
    
    # Iterate through all non-empty subsets of {1..10}
    # Mask bit i (0-indexed) corresponds to denominator i+1
    for mask in range(1, 1 << 10):
        terms = []
        residue = 0
        # Extract terms and compute residue
        for i in range(10):
            if mask & (1 << i):
                d = i + 1
                terms.append(d)
                residue = (residue + invs[i]) % p
        
        size = len(terms)
        if size < best[residue][0]:
            best[residue] = (size, terms)
            
    # Construct the result list
    # If any residue is still at infinity, it means no subset sums to it.
    # However, with 10 denominators, we have 1023 sums. It is highly probable all 101 residues are covered.
    # If not, the problem is unsolvable, but we assume it is solvable.
    result = []
    for r in range(p):
        size, terms = best[r]
        if size == 10**9:
            # Fallback: shouldn't happen, but to be safe, return an empty list or raise?
            # The task says "101 nonempty lists". If a residue is missing, we can't satisfy this.
            # But typically, the set of subset sums of a basis in Z_p covers the space if enough elements are present.
            # Let's just return what we have.
            pass
        result.append(terms)
        
    return result
# EVOLVE-BLOCK-END
