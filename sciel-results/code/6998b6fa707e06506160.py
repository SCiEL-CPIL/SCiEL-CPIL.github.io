"""Improved program for minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random


def _isosceles(a, b, c):
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1 == d2 or d1 == d3 or d2 == d3


def _is_maximal(chosen):
    """Check if the chosen set is maximal: every unchosen point forms an isosceles with some pair in chosen."""
    chosen_set = set(chosen)
    all_points = set(itertools.product(range(N), repeat=2))
    unchosen = all_points - chosen_set
    if not unchosen:
        return True
    for p in unchosen:
        blocked = False
        for a, b in itertools.combinations(chosen, 2):
            if _isosceles(p, a, b):
                blocked = True
                break
        if not blocked:
            return False
    return True


def construct():
    # Step 1: Greedy construction in row-major order
    chosen = []
    for p in itertools.product(range(N), repeat=2):
        if all(not _isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    
    # Step 2: Local search - try to remove points while maintaining maximality
    improved = True
    while improved:
        improved = False
        # Try removing each point in reverse order (last added first, as they might be less critical)
        for i in range(len(chosen) - 1, -1, -1):
            candidate = chosen[:i] + chosen[i+1:]
            if _is_maximal(candidate):
                chosen = candidate
                improved = True
                break  # Restart from beginning after a removal
    
    return chosen
# EVOLVE-BLOCK-END
