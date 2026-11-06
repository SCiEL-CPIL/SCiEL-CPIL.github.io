"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s wall-clock kill
    
    # Precompute modular inverses of 1..10 mod 101
    p = 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # For each residue, track the minimum subset size found and the corresponding subset
    min_len = [11] * p
    best_subsets = [None] * p
    
    # Enumerate all non-empty subsets of {1,...,10}
    # There are only 2^10 - 1 = 1023 subsets, so full enumeration is fast
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
        # Calculate size and modular sum for this subset
        size = 0
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                size += 1
                s += inv[x]
        s %= p
        
        # Update if this subset is smaller than what we have for residue s
        if size < min_len[s]:
            min_len[s] = size
            # Reconstruct the subset
            subset = [x for x in range(1, 11) if mask & (1 << (x - 1))]
            best_subsets[s] = subset
    
    # Safety: if any residue still missing (shouldn't happen with 1023 subsets for 101 residues)
    if any(best_subsets[r] is None for r in range(p)):
        # Fallback: use full set or any valid subset
        all_terms = list(range(1, 11))
        for r in range(p):
            if best_subsets[r] is None:
                best_subsets[r] = all_terms
    
    return [best_subsets[r] for r in range(p)]
# EVOLVE-BLOCK-END
