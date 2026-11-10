import time
from collections import deque
from itertools import combinations

def construct():
    # Precompute modular inverses of 1..10 mod 101
    MOD = 101
    inv = {x: pow(x, -1, MOD) for x in range(1, 11)}
    
    # Generate all subsets of {1..10} and their sizes and residues
    # subsets: list of (size, residue, subset_list)
    subsets = []
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            res = sum(inv[x] for x in combo) % MOD
            subsets.append((s, res, list(combo)))
    
    # For each residue, collect all subsets that achieve it, grouped by size
    # residue_to_subsets[r] = list of (size, subset_list)
    residue_to_subsets = [[] for _ in range(MOD)]
    for s, res, sub in subsets:
        residue_to_subsets[res].append((s, sub))
    
    # For each k from 1 to 10, check if there's a valid assignment
    # We want to assign each residue r a subset of size <= k
    # such that all 101 residues get a distinct subset
    # (distinctness is automatic: if two residues used the same subset, they'd be equal)
    
    # Actually, we just need: for each residue r, pick one subset from residue_to_subsets[r] 
    # with size <= k. Since each subset maps to exactly one residue, different residues 
    # automatically get different subsets. So this is just checking that every residue 
    # has at least one subset of size <= k.
    
    # Wait, that's not right either. The constraint is that denominators are distinct WITHIN a row.
    # But different rows CAN reuse the same denominators. So there's no global distinctness constraint 
    # on subsets across rows. Each row just needs a valid subset.
    
    # So the problem is simply: for each residue r, find the minimum size subset that sums to r mod 101.
    # The answer is the maximum of these minimum sizes.
    
    min_size = [11] * MOD
    for r in range(MOD):
        for s, sub in residue_to_subsets[r]:
            if s < min_size[r]:
                min_size[r] = s
    
    max_row_len = max(min_size)
    
    # Build the result: for each residue, pick a subset of minimum size
    result = []
    for r in range(MOD):
        best_size = min_size[r]
        # Find the first subset of that size
        for s, sub in residue_to_subsets[r]:
            if s == best_size:
                result.append(sub)
                break
    
    return result
