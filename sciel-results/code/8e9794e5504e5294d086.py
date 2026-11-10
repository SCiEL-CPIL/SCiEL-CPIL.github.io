import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # Precompute all subsets and their residues, grouped by size
    # subsets_by_size[s] = list of (residue, subset)
    subsets_by_size = [[] for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            res = 0
            for x in combo:
                res += inv[x]
            res %= 101
            subsets_by_size[s].append((res, list(combo)))
    
    # For each size, build a dict: residue -> one example subset
    example_by_size = [dict() for _ in range(11)]
    for s in range(1, 11):
        for res, subset in subsets_by_size[s]:
            if res not in example_by_size[s]:
                example_by_size[s][res] = subset
    
    # For each residue, find the smallest size s such that r is achievable
    # This gives the optimal assignment since rows are independent
    result = []
    for r in range(101):
        found = None
        for s in range(1, 11):
            if r in example_by_size[s]:
                found = example_by_size[s][r]
                break
        if found is None:
            # Should not happen, but fallback
            found = [1]
        result.append(found)
    
    return result
