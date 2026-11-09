"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute all 4-subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n_all = len(all_sets)  # 1820
    
    # Precompute pairwise intersection sizes
    # inter[i][j] = |all_sets[i] & all_sets[j]|
    inter = [[0] * n_all for _ in range(n_all)]
    for i in range(n_all):
        si = all_sets[i]
        inter[i][i] = 4
        for j in range(i + 1, n_all):
            v = len(si & all_sets[j])
            inter[i][j] = v
            inter[j][i] = v
    
    def is_sunflower_triple(i, j, k):
        """Check if sets i, j, k form a sunflower (all pairwise intersections equal)."""
        a = inter[i][j]
        b = inter[i][k]
        c = inter[j][k]
        return a == b == c
    
    def greedy_random(order):
        """Greedy: process sets in given order, add if no sunflower with two already chosen."""
        chosen = []
        chosen_indices = []
        for idx in order:
            # Check against all pairs in chosen
            ok = True
            for ai in range(len(chosen_indices)):
                for bi in range(ai + 1, len(chosen_indices)):
                    if is_sunflower_triple(idx, chosen_indices[ai], chosen_indices[bi]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(all_sets[idx])
                chosen_indices.append(idx)
        return chosen
    
    best_family = []
    
    # Multiple randomized restarts
    rng = random.Random(42)
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random permutation of all sets
        order = list(range(n_all))
        rng.shuffle(order)
        
        family = greedy_random(order)
        if len(family) > len(best_family):
            best_family = family
        
        # Also try: sort by some heuristic first, then randomize
        # Try ordering by number of conflicts (degree in conflict graph)
        if time.time() > deadline:
            break
    
    # Try a few more with different seeds if time allows
    for r in range(num_restarts, num_restarts + 20):
        if time.time() > deadline:
            break
        order = list(range(n_all))
        rng.shuffle(order)
        family = greedy_random(order)
        if len(family) > len(best_family):
            best_family = family
    
    if not best_family:
        # Fallback: just return a small valid family
        best_family = [set(c) for c in combinations(range(1, 5), 4)]
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
