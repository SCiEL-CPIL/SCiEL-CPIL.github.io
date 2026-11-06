"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # For each residue, find the minimum-size subset that produces it
    best_subset = {}  # residue -> (size, terms_list)
    
    # Iterate by subset size to find minimum size for each residue
    for size in range(1, 11):
        if time.time() > deadline - 5:
            break
        # Generate all subsets of this size
        from itertools import combinations
        for terms in combinations(range(1, 11), size):
            residue = sum(inv[x - 1] for x in terms) % 101
            if residue not in best_subset:
                best_subset[residue] = (size, list(terms))
    
    # If some residues weren't covered (shouldn't happen), fill in with any subset
    if len(best_subset) < 101:
        for mask in range(1, 1 << 10):
            if time.time() > deadline - 2:
                break
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            residue = sum(inv[x - 1] for x in terms) % 101
            if residue not in best_subset:
                best_subset[residue] = (len(terms), terms)
    
    # Build the result: row r contains the terms for residue r
    result = []
    for r in range(101):
        if r in best_subset:
            result.append(best_subset[r][1])
        else:
            # Fallback: use denominator 1 (inverse is 1)
            result.append([1])
    
    return result
# EVOLVE-BLOCK-END
