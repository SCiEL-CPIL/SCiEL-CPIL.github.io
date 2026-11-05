"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # Enumerate all non-empty subsets, group by residue
    # best[r] = list with minimum size for residue r
    best = [None] * 101
    best_size = [11] * 101  # larger than any possible size
    
    for mask in range(1, 1 << 10):
        # Compute residue and size for this subset
        residue = 0
        size = 0
        terms = []
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                residue = (residue + inv[x]) % 101
                size += 1
                terms.append(x)
        
        if size < best_size[residue]:
            best_size[residue] = size
            best[residue] = terms
    
    # Safety: if any residue is uncovered (shouldn't happen), fill with any valid subset
    # This is a fallback; in practice all residues are covered
    if any(b is None for b in best):
        # Fallback: use seed approach
        representations = {}
        for mask in range(1, 1 << 10):
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            residue = sum(inv[x] for x in terms) % 101
            representations.setdefault(residue, terms)
        return [representations[r] for r in range(101)]
    
    return best

# EVOLVE-BLOCK-END
