"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80

    p = 101
    # Precompute inverses for 1..10 modulo 101
    inverses = [pow(x, -1, p) for x in range(1, 11)]

    # Enumerate all 2^10 - 1 = 1023 non-empty subsets of {1, ..., 10}
    # For each subset, compute the sum of inverses mod 101 and the length
    # Build a mapping: residue -> list of (length, terms)
    residue_to_subsets = defaultdict(list)
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                terms.append(denom)
                s += inverses[i]
        residue = s % p
        length = len(terms)
        residue_to_subsets[residue].append((length, terms))

    # For each residue, sort its subsets by length (ascending)
    # and pick the shortest one. This minimizes the maximum row length.
    # Since denominators can recur across rows, we just need one valid
    # subset per residue.
    
    result = [None] * p
    max_len = 0
    for r in range(p):
        if r in residue_to_subsets:
            # Sort by length, pick the shortest
            subs = residue_to_subsets[r]
            subs.sort(key=lambda x: x[0])
            best_len, best_terms = subs[0]
            result[r] = best_terms
            if best_len > max_len:
                max_len = best_len
        else:
            # Should not happen since all 101 residues are covered
            result[r] = [1]
            max_len = max(max_len, 1)

    return result
# EVOLVE-BLOCK-END
