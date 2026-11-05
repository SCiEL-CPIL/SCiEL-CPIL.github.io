import time
from collections import defaultdict
import heapq

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    P = 101
    N_DENOMS = 10
    
    # Precompute inverses
    inv = [0] * (N_DENOMS + 1)
    for x in range(1, N_DENOMS + 1):
        inv[x] = pow(x, -1, P)
    
    # Precompute all non-empty subsets of {1..10} with their residue and size
    # There are 2^10 - 1 = 1023 subsets
    all_subsets = []  # list of (residue, size, subset_list)
    for mask in range(1, 1 << N_DENOMS):
        subset = []
        s = 0
        for x in range(1, N_DENOMS + 1):
            if mask >> (x - 1) & 1:
                subset.append(x)
                s += inv[x]
        r = s % P
        sz = len(subset)
        all_subsets.append((r, sz, subset))
    
    # Group subsets by residue
    subsets_by_residue = defaultdict(list)
    for r, sz, subset in all_subsets:
        subsets_by_residue[r].append((sz, subset))
    
    # For each residue, sort by size (smallest first)
    for r in subsets_by_residue:
        subsets_by_residue[r].sort(key=lambda x: x[0])
    
    # Strategy: Minimize the maximum row length.
    # Try max_size from 1 to 10. For each max_size, check if we can assign
    # a subset of size <= max_size to each of the 101 residues.
    # This is equivalent to: for each residue r, is there at least one subset
    # of size <= max_size that sums to r mod P?
    # Since subsets can be reused across different residues (denominators may recur
    # across different rows), we just need each residue to have at least one valid
    # subset of size <= max_size.
    
    best_assignment = None
    for max_size in range(1, N_DENOMS + 1):
        if time.time() > deadline:
            break
        
        assignment = {}
        feasible = True
        for r in range(P):
            found = False
            for sz, subset in subsets_by_residue.get(r, []):
                if sz <= max_size:
                    assignment[r] = subset
                    found = True
                    break
            if not found:
                feasible = False
                break
        
        if feasible:
            best_assignment = assignment
            break
    
    if best_assignment is None:
        # Fallback: use the minimum-size subset for each residue
        best_assignment = {}
        for r in range(P):
            if r in subsets_by_residue:
                best_assignment[r] = subsets_by_residue[r][0][1]
            else:
                # Should not happen, but ensure non-empty list
                best_assignment[r] = [1]
    
    return [best_assignment[r] for r in range(P)]
# EVOLVE-BLOCK-END
