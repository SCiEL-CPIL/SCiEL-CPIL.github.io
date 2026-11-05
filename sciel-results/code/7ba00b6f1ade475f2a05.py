"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # Generate all non-empty subsets of {1,...,10} with their residue and size
    # Group by size to enable early stopping
    subsets_by_size = {k: [] for k in range(1, 11)}
    for k in range(1, 11):
        for combo in combinations(range(1, 11), k):
            residue = sum(inv[x] for x in combo) % 101
            subsets_by_size[k].append((residue, list(combo)))
    
    # For each residue, we want the smallest subset that covers it
    # We'll do a bipartite matching: residues -> subsets, minimizing max size
    # Strategy: try k=1,2,... and find the minimum k such that all 101 residues
    # can be covered by subsets of size <= k
    
    # For a given max_size k, check if we can assign to each residue a subset
    # of size <= k such that all residues are covered.
    # Since denominators can recur across rows, this is just: for each residue,
    # pick any subset of size <= k that maps to it.
    # So the condition is simply: for every residue r in 0..100, there exists
    # at least one subset of size <= k whose sum of inverses mod 101 equals r.
    
    # Find minimum k
    for k in range(1, 11):
        covered = set()
        for size in range(1, k + 1):
            for residue, terms in subsets_by_size[size]:
                covered.add(residue)
        if len(covered) == 101:
            break
    
    # Now assign to each residue the smallest subset (by size, then lexicographic)
    best_for_residue = {}
    for size in range(1, k + 1):
        for residue, terms in subsets_by_size[size]:
            if residue not in best_for_residue:
                best_for_residue[residue] = terms
    
    # Verify all residues are covered
    assert len(best_for_residue) == 101, f"Only {len(best_for_residue)} residues covered"
    
    # Build the result
    result = [best_for_residue[r] for r in range(101)]
    
    # Check time
    if time.time() > deadline:
        # Should not happen since this is fast, but guard just in case
        pass
    
    return result
# EVOLVE-BLOCK-END
