import time
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 85.0  # Safety margin

    P = 101
    D = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
    
    # Build all nonempty subsets with their residues, grouped by size
    # We store (residue, size, terms)
    subsets_by_size = {i: [] for i in range(1, 11)}
    
    for size in range(1, 11):
        for combo in combinations(D, size):
            res = sum(inv[x] for x in combo) % P
            subsets_by_size[size].append((res, size, list(combo)))
    
    # Find minimum k such that subsets of size <= k cover all 101 residues
    best_k = None
    for k in range(1, 11):
        covered = set()
        for s in range(1, k + 1):
            for res, _, _ in subsets_by_size[s]:
                covered.add(res)
        if len(covered) == 101:
            best_k = k
            break
    
    if best_k is None:
        # Fallback: should not happen for P=101, D=1..10
        best_k = 10
        
    # Check deadline before assignment
    if time.time() - t0 > deadline:
        return [list(range(1, 11)) for _ in range(101)]

    # Build map: residue -> list of (size, terms) for sizes <= best_k
    # We want to pick the smallest size subset for each residue to be safe,
    # although any valid subset of size <= best_k works for the max constraint.
    residue_to_best_subset = {r: None for r in range(101)}
    
    # Iterate by size to ensure we find the smallest size first
    for s in range(1, best_k + 1):
        for res, size, terms in subsets_by_size[s]:
            if residue_to_best_subset[res] is None:
                residue_to_best_subset[res] = (size, terms)
            # If we already have a subset of smaller size, we don't need to update
            # because we iterate s from 1 upwards.
    
    assignment = []
    for r in range(101):
        if residue_to_best_subset[r]:
            _, terms = residue_to_best_subset[r]
            assignment.append(terms)
        else:
            # Should not happen if best_k was found correctly
            assignment.append([1])
            
    return assignment

# EVOLVE-BLOCK-END
