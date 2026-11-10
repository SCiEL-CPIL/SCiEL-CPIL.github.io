import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # For each size s from 1 to 10, compute all residues achievable
    # and store one example subset for each residue
    # residues_by_size[s] = dict mapping residue -> subset (first found)
    residues_by_size = [dict() for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            residue = sum(inv[x] for x in combo) % 101
            if residue not in residues_by_size[s]:
                residues_by_size[s][residue] = list(combo)
    
    # Find the smallest max_size such that all 101 residues are achievable
    # with subsets of size <= max_size
    min_max_size = None
    for max_s in range(1, 11):
        all_covered = True
        for r in range(101):
            found = False
            for s in range(1, max_s + 1):
                if r in residues_by_size[s]:
                    found = True
                    break
            if not found:
                all_covered = False
                break
        if all_covered:
            min_max_size = max_s
            break
    
    if min_max_size is None:
        min_max_size = 10  # should not happen
    
    # Assign each residue to the smallest subset size <= min_max_size that achieves it
    result = []
    for r in range(101):
        for s in range(1, min_max_size + 1):
            if r in residues_by_size[s]:
                result.append(residues_by_size[s][r])
                break
        else:
            # Fallback: use all denominators
            result.append(list(range(1, 11)))
    
    # Verify correctness
    for r in range(101):
        row = result[r]
        assert len(row) > 0, f"Row {r} is empty"
        assert len(set(row)) == len(row), f"Row {r} has duplicate denominators"
        assert all(1 <= x <= 10 for x in row), f"Row {r} has out-of-range denominators"
        total = sum(inv[x] for x in row) % 101
        assert total == r, f"Row {r}: sum of inverses is {total}, expected {r}"
    
    return result
