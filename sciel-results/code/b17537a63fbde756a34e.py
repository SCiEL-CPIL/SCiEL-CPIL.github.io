"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations

def check_element(a, others):
    """Check if 1/a equals sum of reciprocals of any non-empty subset of others."""
    n = len(others)
    for mask in range(1, 1 << n):
        total = 0.0
        for i in range(n):
            if mask & (1 << i):
                total += 1.0 / others[i]
        if abs(total - 1.0 / a) < 1e-12:
            return True
    return False

def is_valid(S):
    """Check if set S satisfies the reciprocal-subsum-free property."""
    for a in S:
        others = [x for x in S if x != a]
        if check_element(a, others):
            return False
    return True

def construct():
    # Start with all elements, remove those that are subsumed
    S = list(range(1, 25))
    
    # Iteratively remove elements that violate the constraint
    changed = True
    while changed:
        changed = False
        for a in S[:]:
            others = [x for x in S if x != a]
            if check_element(a, others):
                S.remove(a)
                changed = True
    
    return S
# EVOLVE-BLOCK-END
