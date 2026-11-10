"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import lcm
from itertools import combinations

# EVOLVE-BLOCK-START
import time
from math import lcm

def _lcm_1_to_22():
    result = 1
    for i in range(1, 23):
        result = lcm(result, i)
    return result

_LCM = _lcm_1_to_22()

def _is_dissociated(indices):
    """Check if the set of indices (1-based) has all distinct reciprocal subset sums.
    Uses integer scaling by LCM(1..22)."""
    if not indices:
        return True
    n = len(indices)
    # Integer weights: weight[i] = LCM // indices[i]
    weights = [_LCM // idx for idx in indices]
    
    # Generate all 2^n subset sums
    # For n <= 20, this is at most 1 million, which is fast
    # Use iterative approach
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        # Check for collision
        for ns in new_sums:
            if ns in sums:
                return False
        sums |= new_sums
    
    # All 2^n sums are distinct
    return len(sums) == (1 << n)

def _greedy_build(order, deadline):
    """Greedy: try to add elements in the given order, keeping the set if dissociated."""
    current = []
    for idx in order:
        if time.time() > deadline:
            break
        candidate = current + [idx]
        if _is_dissociated(candidate):
            current = candidate
    return current

def construct():
    deadline = time.time() + 80  # 80 second deadline (20s safety margin)
    
    # Try multiple orderings
    base = list(range(1, 23))
    
    # Ordering 1: decreasing (largest denominator first = smallest reciprocal)
    order1 = sorted(base, reverse=True)
    
    # Ordering 2: increasing
    order2 = sorted(base)
    
    # Ordering 3: primes first, then composites
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    composites = [1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22]
    order3 = primes + composites
    
    # Ordering 4: alternating large and small
    order4 = []
    large = sorted(base, reverse=True)
    small = sorted(base)
    for i in range(22):
        order4.append(large[i])
        if i < 11:
            order4.append(small[i])
    
    # Ordering 5: sorted by 1/k (i.e., same as decreasing k)
    order5 = sorted(base, key=lambda x: 1.0/x)
    
    # Ordering 6: try to maximize spread - start with extremes
    order6 = [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12]
    
    orderings = [order1, order2, order3, order4, order5, order6]
    
    best = []
    for order in orderings:
        if time.time() > deadline:
            break
        result = _greedy_build(order, deadline)
        if len(result) > len(best):
            best = result
    
    # Local search: try to add remaining elements in different orders
    if time.time() < deadline:
        remaining = [x for x in base if x not in best]
        # Try adding remaining elements one by one in various orders
        for attempt_order in [sorted(remaining, reverse=True), sorted(remaining), sorted(remaining, key=lambda x: 1.0/x)]:
            if time.time() > deadline:
                break
            for idx in attempt_order:
                if time.time() > deadline:
                    break
                if idx in best:
                    continue
                candidate = best + [idx]
                if _is_dissociated(candidate):
                    best = candidate
    
    # If best is empty (shouldn't happen), return at least [1]
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
