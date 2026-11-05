"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s/90s limit
    
    P = 101
    DENOMS = list(range(1, 11))
    INV = {x: pow(x, -1, P) for x in DENOMS}
    
    # Precompute inverse values for each denominator
    inv_vals = [INV[x] for x in DENOMS]
    
    # Precompute all subset sums organized by (size, residue)
    # subsets_by_size_residue[size][residue] = list of term-lists
    subsets_by_size_residue = defaultdict(lambda: defaultdict(list))
    
    # Also track all subsets for fallback
    all_subsets = []
    
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        terms = [DENOMS[i] for i in range(10) if (mask >> i) & 1]
        residue = sum(inv_vals[i] for i in range(10) if (mask >> i) & 1) % P
        subsets_by_size_residue[size][residue].append(terms)
        all_subsets.append((residue, terms))
    
    # For each max_size, build a quick lookup: residue -> list of terms with size <= max_size
    # We'll check feasibility incrementally
    
    # Build cumulative structures: for each k, which residues have at least one subset of size <= k
    # and store the smallest such subset per residue
    
    best_assignment = None
    for max_size in range(1, 11):
        if time.time() > deadline:
            break
        
        # Check if all 101 residues are covered by subsets of size <= max_size
        assignment = {}
        feasible = True
        for r in range(P):
            # Find smallest-size subset for this residue with size <= max_size
            found = None
            for sz in range(1, max_size + 1):
                if r in subsets_by_size_residue[sz]:
                    found = subsets_by_size_residue[sz][r][0]
                    break
            if found is None:
                feasible = False
                break
            assignment[r] = found
        
        if feasible:
            best_assignment = assignment
            break
    
    if best_assignment is None:
        # Fallback: use any representation
        representations = {}
        for res, terms in all_subsets:
            if res not in representations:
                representations[res] = terms
        best_assignment = representations
    
    # Convert to list format
    result = [best_assignment[r] for r in range(P)]
    return result

# EVOLVE-BLOCK-END
