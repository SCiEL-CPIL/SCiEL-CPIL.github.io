"""Deterministic greedy with multiple structured orderings."""
# EVOLVE-BLOCK-START
import itertools

def _pairwise_intersections_equal(a, b, c):
    """Check if three distinct 4-sets have all three pairwise intersections equal."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _is_sunflower_free(chosen, new_set):
    """Check if adding new_set to chosen creates a sunflower."""
    for i in range(len(chosen)):
        a = chosen[i]
        for j in range(i + 1, len(chosen)):
            b = chosen[j]
            if _pairwise_intersections_equal(a, b, new_set):
                return False
    return True

def _greedy_deterministic(ordering):
    """Greedy with a specific deterministic ordering of all 4-subsets."""
    all_4sets = list(itertools.combinations(range(1, 17), 4))
    if ordering == "sum":
        all_4sets.sort(key=lambda s: sum(s))
    elif ordering == "binary":
        all_4sets.sort(key=lambda s: int(''.join(str(x % 2) for x in s), 2) if len(s) == 4 else 0)
    elif ordering == "lex":
        all_4sets.sort()
    elif ordering == "reverse_sum":
        all_4sets.sort(key=lambda s: -sum(s))
    
    chosen = []
    for s in all_4sets:
        s_set = set(s)
        if _is_sunflower_free(chosen, s_set):
            chosen.append(s_set)
    
    return [sorted(s) for s in chosen]

def construct():
    """Construct a large sunflower-free family using deterministic greedy with multiple orderings."""
    best_family = []
    best_size = 0
    
    orderings = ["sum", "reverse_sum", "lex", "binary"]
    for ordering in orderings:
        family = _greedy_deterministic(ordering)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
    
    return best_family
# EVOLVE-BLOCK-END
