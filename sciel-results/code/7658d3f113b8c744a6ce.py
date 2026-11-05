"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # Precompute all non-empty subsets: (residue, size, terms)
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(inv[x] for x in terms) % 101
        size = len(terms)
        all_subsets.append((residue, size, terms))
    
    # For each residue, collect all subsets that achieve it, sorted by size
    from collections import defaultdict
    residue_to_subsets = defaultdict(list)
    for residue, size, terms in all_subsets:
        residue_to_subsets[residue].append((size, terms))
    
    # For each residue, sort by size and keep the smallest
    # We want to minimize the maximum row length
    # Find minimum k such that all residues have at least one subset of size <= k
    min_k = 10
    for k in range(1, 11):
        all_covered = True
        for r in range(101):
            if not any(size <= k for size, terms in residue_to_subsets[r]):
                all_covered = False
                break
        if all_covered:
            min_k = k
            break
    
    # Build the result: for each residue, pick the smallest-size subset (or any with size <= min_k)
    result = []
    for r in range(101):
        # Pick the subset with minimum size (tie-break: first found)
        best = min(residue_to_subsets[r], key=lambda x: x[0])
        result.append(best[1])
    
    return result
# EVOLVE-BLOCK-END
