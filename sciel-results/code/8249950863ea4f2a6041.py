"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill
    
    MOD = 101
    DENOMS = list(range(1, 11))
    N = len(DENOMS)
    
    # Precompute inverses
    invs = [pow(x, -1, MOD) for x in DENOMS]
    
    # min_len_for_residue[r] will store the minimum length of a subset summing to r
    # best_subset_for_residue[r] will store the actual subset (as a list of denominators)
    min_len_for_residue = [11] * MOD
    best_subset_for_residue = [None] * MOD
    
    # Iterate over all non-empty subsets of denominators
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << N):
        terms = []
        s = 0
        for i in range(N):
            if mask & (1 << i):
                terms.append(DENOMS[i])
                s += invs[i]
        
        s %= MOD
        length = len(terms)
        
        # Update if this subset is shorter than the current best for this residue
        if length < min_len_for_residue[s]:
            min_len_for_residue[s] = length
            best_subset_for_residue[s] = terms
        
        # Optional: Early termination if we've found length 1 for all residues?
        # Unlikely to be efficient to check, and max length is likely > 1.
        
    # Build the result
    result = []
    for r in range(MOD):
        if best_subset_for_residue[r] is not None:
            result.append(list(best_subset_for_residue[r]))
        else:
            # Fallback: This should not happen if the problem is solvable.
            # If it does, we return an empty list, which will likely fail validation.
            result.append([])
            
    return result
# EVOLVE-BLOCK-END
