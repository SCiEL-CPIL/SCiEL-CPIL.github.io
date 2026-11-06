"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct 101 nonempty lists, row r containing distinct denominators in 1..10
    whose modular inverses sum to r modulo 101. Minimize the largest row length.
    
    Strategy: For each residue, find the smallest subset (by size) of {1..10}
    whose modular inverses sum to that residue. This minimizes max row length.
    """
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inverses = [0] * 11  # 1-indexed
    for x in range(1, 11):
        inverses[x] = pow(x, -1, 101)
    
    # For each residue 0..100, find the minimum-size subset that sums to it
    # There are 2^10 - 1 = 1023 non-empty subsets, so this is very fast
    
    # best_terms[r] = list of denominators with minimum size summing to r
    best_terms = [None] * 101
    best_size = [float('inf')] * 101
    
    # Iterate over all non-empty subsets of {1, ..., 10}
    # For each subset, compute the residue and size
    # Keep track of the smallest size for each residue
    for mask in range(1, 1 << 10):
        # Compute the terms and their sum
        terms = []
        s = 0
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                terms.append(x)
                s = (s + inverses[x]) % 101
        
        size = len(terms)
        if size < best_size[s]:
            best_size[s] = size
            best_terms[s] = terms[:]
    
    # Check if all residues are covered
    for r in range(101):
        if best_terms[r] is None:
            # This shouldn't happen with 1023 subsets and 101 residues,
            # but handle it gracefully
            raise ValueError(f"Residue {r} not covered")
    
    # The result is best_terms[r] for r in 0..100
    return best_terms

# EVOLVE-BLOCK-END
