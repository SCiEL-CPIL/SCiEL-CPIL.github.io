import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    p = 101
    denominators = list(range(1, 11))
    invs = [pow(x, -1, p) for x in denominators]
    
    # Precompute all subset sums (all 2^10 - 1 non-empty subsets)
    # Map: residue -> list of (subset_size, subset_tuple)
    residue_to_subsets = {r: [] for r in range(p)}
    
    for k in range(1, 11):
        for combo in combinations(denominators, k):
            s = 0
            for x in combo:
                s += invs[x - 1]
            s %= p
            residue_to_subsets[s].append((k, combo))
    
    # Check if k=3 covers all residues
    # If so, that's the minimum (k=1 has at most 10 residues, k=2 has at most C(10,2)=45)
    # C(10,3) = 120 > 101, so k=3 might work
    
    # For each residue, pick the smallest-size subset available
    # This gives us the minimal max row length
    assignment = {}
    for r in range(p):
        if residue_to_subsets[r]:
            # Sort by size, then by the subset itself for determinism
            residue_to_subsets[r].sort(key=lambda x: (x[0], x[1]))
            # Pick the smallest
            k, combo = residue_to_subsets[r][0]
            assignment[r] = list(combo)
    
    # Determine the max row length
    if assignment:
        max_len = max(len(v) for v in assignment.values())
    else:
        max_len = 10
    
    # Build result
    result = []
    for r in range(p):
        if r in assignment:
            result.append(assignment[r])
        else:
            # Fallback: find any subset that sums to r
            found = False
            for k in range(1, 11):
                for combo in combinations(denominators, k):
                    s = sum(invs[x - 1] for x in combo) % p
                    if s == r:
                        result.append(list(combo))
                        found = True
                        break
                if found:
                    break
            if not found:
                result.append([1])
    
    return result
