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
    
    # Build all nonempty subsets with their residues, grouped by size
    subsets_by_size = {i: [] for i in range(1, 11)}
    
    for size in range(1, 11):
        for combo in combinations(D, size):
            res = sum(inv[x] for x in combo) % P
            subsets_by_size[size].append((res, list(combo)))
    
    # Find minimum k such that subsets of size <= k cover all 101 residues
    best_k = None
    for k in range(1, 11):
        covered = set()
        for s in range(1, k + 1):
            for res, _ in subsets_by_size[s]:
                covered.add(res)
        if len(covered) == 101:
            best_k = k
            break
    
    if best_k is None:
        best_k = 10
        
    # Check deadline before assignment
    if time.time() - t0 > deadline:
        # Fallback: return all 10s (invalid but safe structure)
        return [list(range(1, 11)) for _ in range(101)]

    # Build map: residue -> list of (size, terms) for sizes <= best_k
    residue_to_subsets = {r: [] for r in range(101)}
    for s in range(1, best_k + 1):
        for res, terms in subsets_by_size[s]:
            residue_to_subsets[res].append((s, terms))
    
    assignment = [None] * 101
    for r in range(101):
        if residue_to_subsets[r]:
            _, terms = residue_to_subsets[r][0]
            assignment[r] = terms
        else:
            assignment[r] = [1]
            
    return assignment

# EVOLVE-BLOCK-END
