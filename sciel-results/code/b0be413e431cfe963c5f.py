"""Search-based program: find minimum max row length covering all residues mod 101."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    P = 101
    D = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
    
    # For each residue, find the minimum subset size that achieves it
    min_size_for_residue = [11] * P  # initialize to impossible
    
    # Enumerate all non-empty subsets, track minimum size per residue
    for size in range(1, 11):
        for combo in combinations(D, size):
            res = sum(inv[x] for x in combo) % P
            if size < min_size_for_residue[res]:
                min_size_for_residue[res] = size
    
    # Find the maximum minimum size across all residues
    max_min_size = max(min_size_for_residue)
    
    # Build assignment: for each residue, use a subset of minimum size
    # We need to store which subset achieves the minimum
    best_subset_for_residue = [None] * P
    
    for size in range(1, max_min_size + 1):
        for combo in combinations(D, size):
            res = sum(inv[x] for x in combo) % P
            if best_subset_for_residue[res] is None:
                best_subset_for_residue[res] = list(combo)
    
    # Check deadline
    if time.time() - t0 > deadline:
        return [list(range(1, 11)) for _ in range(101)]
    
    # Build final assignment
    assignment = []
    for r in range(P):
        if best_subset_for_residue[r] is not None:
            assignment.append(best_subset_for_residue[r])
        else:
            # Should not happen, but fallback
            assignment.append([1])
    
    return assignment

# EVOLVE-BLOCK-END
