"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_valid(selected):
    """Check if a set of selected integers satisfies the no-dividing condition."""
    n = len(selected)
    if n < 4:
        return True
    # For each element a in selected, check that a does not divide the sum of any 3 distinct other elements
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _greedy_construct(order):
    """Greedy construction using a given order of candidates."""
    selected = []
    for x in order:
        candidate = selected + [x]
        if _is_valid(candidate):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Try multiple orders to find a better solution
    best = []
    
    # Order 1: Descending (original approach)
    if time.time() < deadline:
        order1 = list(range(192, 0, -1))
        result1 = _greedy_construct(order1)
        if len(result1) > len(best):
            best = result1
    
    # Order 2: Only numbers > 96 (to avoid small divisors)
    if time.time() < deadline:
        order2 = list(range(192, 96, -1))
        result2 = _greedy_construct(order2)
        if len(result2) > len(best):
            best = result2
    
    # Order 3: Numbers > 64
    if time.time() < deadline:
        order3 = list(range(192, 64, -1))
        result3 = _greedy_construct(order3)
        if len(result3) > len(best):
            best = result3
    
    # Order 4: Numbers > 48
    if time.time() < deadline:
        order4 = list(range(192, 48, -1))
        result4 = _greedy_construct(order4)
        if len(result4) > len(best):
            best = result4
    
    # Order 5: Try to add smaller numbers to the best set found so far
    if time.time() < deadline and len(best) < 48:
        # Sort best in ascending order for checking
        best_sorted = sorted(best)
        # Try to add numbers from 192 down to 1 that aren't already in best
        for x in range(192, 0, -1):
            if x in best:
                continue
            candidate = best + [x]
            if _is_valid(candidate):
                best = candidate
            if time.time() > deadline:
                break
    
    return best
# EVOLVE-BLOCK-END
