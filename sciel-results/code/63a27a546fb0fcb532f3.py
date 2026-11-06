import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute inverses of 1..10 modulo 101
    invs = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all subsets (non-empty) with their sums
    # subsets_by_size[k] = list of (sum, comb)
    subsets_by_size = {k: [] for k in range(1, 11)}
    
    for k in range(1, 11):
        for comb in combinations(range(1, 11), k):
            s = sum(invs[x-1] for x in comb) % 101
            subsets_by_size[k].append((s, comb))
            
    # Strategy: Binary search for the minimal K such that all residues are covered by subsets of size <= K.
    # This aligns with the "epigraph" formulation: minimize K subject to coverage constraints.
    
    def is_feasible(K):
        # Check if subsets of size <= K cover all 101 residues
        covered = set()
        for k in range(1, K + 1):
            for s, comb in subsets_by_size[k]:
                covered.add(s)
                if len(covered) == 101:
                    return True
        return len(covered) == 101

    # Find minimal K
    lo, hi = 1, 10
    min_K = 10
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_feasible(mid):
            min_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Now, for each residue, find the subset of size <= min_K with the minimum size.
    # We iterate by size to ensure we pick the smallest one first.
    best_rep = {}
    for k in range(1, min_K + 1):
        for s, comb in subsets_by_size[k]:
            if s not in best_rep:
                best_rep[s] = comb
            if len(best_rep) == 101:
                break
        if len(best_rep) == 101:
            break
            
    # Handle any missing residues (should not happen if is_feasible was correct, but for safety)
    if len(best_rep) < 101:
        for r in range(101):
            if r not in best_rep:
                # Fallback: search all sizes
                for k in range(1, 11):
                    for s, comb in subsets_by_size[k]:
                        if s == r:
                            best_rep[r] = comb
                            break
                    if r in best_rep:
                        break
                if r not in best_rep:
                    best_rep[r] = [1]

    return [best_rep[r] for r in range(101)]
