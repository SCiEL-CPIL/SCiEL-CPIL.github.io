import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute modular inverses of 1..10 mod 101
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # For each size s from 1 to 10, compute all residues achievable
    # and store one example subset for each residue
    # residues_by_size[s] = dict mapping residue -> subset (first found)
    residues_by_size = [dict() for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            residue = sum(inv[x] for x in combo) % 101
            if residue not in residues_by_size[s]:
                residues_by_size[s][residue] = list(combo)
    
    # Now we need to assign each residue 0..100 to a row with distinct denominators
    # We want to minimize the largest row length.
    # Each row r must contain distinct denominators whose inverses sum to r mod 101.
    # The denominators in row r are a subset of {1..10}.
    
    # For each residue r, find the minimum size s such that there exists a subset
    # of size s with sum of inverses = r mod 101.
    # But we also need all rows to have distinct denominators *within* each row,
    # which is already guaranteed since each row is a subset.
    
    # The key constraint: each row r has a set of denominators (subset of 1..10)
    # such that the sum of their inverses ≡ r (mod 101).
    # Different rows can reuse the same denominators.
    # We want to minimize the maximum row length.
    
    # For each residue r, find the minimum size subset that achieves it.
    min_size_for_residue = {}
    for r in range(101):
        min_size_for_residue[r] = 11  # default: not found
        for s in range(1, 11):
            if r in residues_by_size[s]:
                min_size_for_residue[r] = s
                break
    
    # The answer is the maximum of min_size_for_residue over all r
    # But wait - we need to actually construct the rows.
    # Each row r gets the minimum-size subset that sums to r mod 101.
    
    result = []
    for r in range(101):
        s = min_size_for_residue[r]
        result.append(residues_by_size[s][r])
    
    return result
