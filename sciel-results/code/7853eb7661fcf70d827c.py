"""Randomized greedy with multiple restarts."""
# EVOLVE-BLOCK-START
import itertools
import random

def _pairwise_intersections_equal(a, b, c):
    """Check if three distinct 4-sets have all three pairwise intersections equal."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _is_sunflower_free(chosen, new_set):
    """Check if adding new_set to chosen creates a sunflower."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        for j in range(i + 1, n):
            b = chosen[j]
            if _pairwise_intersections_equal(a, b, new_set):
                return False
    return True

def _greedy_randomized():
    """Greedy with a random ordering of all 4-subsets."""
    all_4sets = list(itertools.combinations(range(1, 17), 4))
    random.shuffle(all_4sets)
    
    chosen = []
    for s in all_4sets:
        s_set = frozenset(s)
        if _is_sunflower_free(chosen, s_set):
            chosen.append(s_set)
    
    return [sorted(s) for s in chosen]

def construct():
    """Construct a large sunflower-free family using randomized greedy with restarts."""
    best_family = []
    best_size = 0
    
    # Run multiple restarts to escape local optima of the greedy choice
    # 50 restarts should be fast enough (approx 1-2 seconds per restart worst case, but usually faster as family size is limited)
    # Actually, 1820 sets * 50^2 checks is ~4.5M set ops. 50 restarts is 225M set ops.
    # This might be too slow in Python. Let's try 20 restarts first.
    
    num_restarts = 20
    
    for _ in range(num_restarts):
        family = _greedy_randomized()
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            
    return best_family
# EVOLVE-BLOCK-END
