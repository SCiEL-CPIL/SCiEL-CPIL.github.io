from itertools import combinations

def construct():
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n = len(all_sets)
    
    # Precompute sunflower triples: for each i, list of (j,k) with j<k, j,k != i, 
    # such that sets[j], sets[k], sets[i] form a sunflower
    # Actually, let's precompute for each pair (i,j), the set of k such that (i,j,k) is a sunflower
    # Or better: for each i, precompute a bitmask or list of conflicting pairs
    
    # Let's precompute: conflict_pairs[i] = list of (j,k) where j<k, j,k != i, and (i,j,k) is sunflower
    # This is O(n^3 * 4) which is 1820^3 * 4 ≈ 24 billion. Too slow.
    
    # Better approach: for each pair (i,j), compute which k form a sunflower with i and j.
    # For fixed i,j: I = sets[i] & sets[j]. Then k must satisfy:
    # sets[i] & sets[k] = I and sets[j] & sets[k] = I
    # This means:
    #   I ⊆ sets[k]
    #   sets[k] ∩ (sets[i] \ I) = ∅
    #   sets[k] ∩ (sets[j] \ I) = ∅
    #   |sets[k]| = 4
    #   sets[k] != sets[i], sets[k] != sets[j]
    
    # For each pair (i,j), the valid k are 4-sets containing I, disjoint from (sets[i]\I) ∪ (sets[j]\I), 
    # and not equal to sets[i] or sets[j].
    
    # Let's precompute for each pair (i,j) the list of k that form a sunflower.
    # Number of pairs: C(1820,2) ≈ 1.65 million. For each, we need to find valid k.
    # The universe for k is constrained: must contain I (|I| elements), be disjoint from forbidden = (sets[i]\I) ∪ (sets[j]\I),
    # and have size 4. The free elements are [16] \ (sets[i] ∪ sets[j] ∪ I) but wait, I ⊆ sets[i] and I ⊆ sets[j].
    # Actually, free elements = [16] \ (sets[i] ∪ sets[j]). Since I ⊆ sets[i] ∪ sets[j], and we need to pick 4-|I| more from free elements.
    # Number of free elements = 16 - |sets[i] ∪ sets[j]|.
    
    # This is getting complex. Let me just use the on-the-fly approach but with early termination and hope it's fast enough.
    
    # Actually, let me try a different greedy: instead of minimum conflicts, use a randomized or different ordering.
    
    # Let me try: sort all sets by some heuristic and greedily add if no conflict.
    # Or: use a priority queue where priority = number of conflicts.
    
    # For now, let me optimize the on-the-fly check by caching pairwise intersections.
    
    # Precompute all pairwise intersections
    inter = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            inter[i][j] = all_sets[i] & all_sets[j]
            inter[j][i] = inter[i][j]
    
    def is_sunflower_fast(i, j, k):
        """Check if sets i, j, k form a sunflower using precomputed intersections."""
        ij = inter[i][j]
        ik = inter[i][k]
        jk = inter[j][k]
        return ij == ik == jk
    
    chosen = []
    chosen_set = set()
    available = list(range(n))
    
    while available:
        best_idx = -1
        best_conflicts = float('inf')
        
        for idx in available:
            conflicts = 0
            for i in chosen:
                for j in chosen:
                    if i >= j:
                        continue
                    if is_sunflower_fast(i, j, idx):
                        conflicts += 1
                        if conflicts >= best_conflicts:
                            break
                if conflicts >= best_conflicts:
                    break
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_idx = idx
                if conflicts == 0:
                    break
        
        if best_conflicts > 0:
            break
        
        chosen.append(best_idx)
        available.remove(best_idx)
    
    return [sorted(all_sets[i]) for i in chosen]
