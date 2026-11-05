import random
from itertools import combinations

def construct():
    # Precompute all 4-element subsets of [1, 16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n_total = len(all_sets)
    
    # For each pair of sets, precompute their intersection
    # inter[i][j] = all_sets[i] & all_sets[j]
    # To save memory and time, we can store as frozensets or just compute on the fly
    # But for speed in the inner loop, precomputation is key.
    # Let's store intersections in a 2D list. 1820x1820 is ~3.3M entries.
    # Each entry is a frozenset of size 0-4. This might be heavy but manageable.
    # Alternatively, we can check conflicts on the fly but that's O(n^2) per insertion.
    
    # Let's try a greedy approach with randomization and multiple restarts, 
    # but optimize the conflict check.
    
    # Conflict check for adding set S to family F:
    # For all pairs (A, B) in F, check if A&B == A&S == B&S.
    # This is O(|F|^2) per insertion.
    # With |F| up to ~200, that's 40,000 checks per insertion.
    # Total insertions ~1820. Total ops ~70M. This might be too slow for Python in 60s.
    
    # We need a faster check.
    # Note that A&B == A&S == B&S implies that A, B, S have a common core C = A&B = A&S = B&S.
    # Also, A\S, B\S, S\A, S\B are disjoint from C.
    # Actually, the condition is that the three pairwise intersections are equal.
    # Let X = A&B, Y = A&S, Z = B&S. We need X==Y==Z.
    
    # Let's stick to the direct check but optimize with early termination.
    # Also, we can use a better heuristic: try to add sets that are "far" from existing ones.
    
    best_family = []
    
    # Use multiple random seeds and strategies
    for seed in range(100):
        random.seed(seed)
        family = []
        remaining = list(range(n_total))
        random.shuffle(remaining)
        
        for idx in remaining:
            s = all_sets[idx]
            ok = True
            # Check against all pairs in family
            for i in range(len(family)):
                a = family[i]
                a_s = a & s
                for j in range(i+1, len(family)):
                    b = family[j]
                    if a & b == a_s == b & s:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                family.append(s)
        
        if len(family) > len(best_family):
            best_family = family
    
    return [sorted(s) for s in best_family]
