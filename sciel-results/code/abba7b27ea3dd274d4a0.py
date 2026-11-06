import time
from fractions import Fraction
from itertools import combinations

DEADLINE = 85.0
_START = None

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _subset_sum_possible(vals, target):
    n = len(vals)
    if n == 0:
        return target == Fraction(0)
    if n <= 1:
        return any(v == target for v in vals)
    mid = n // 2
    left, right = vals[:mid], vals[mid:]
    L = {Fraction(0)}
    for v in left:
        L = L | {x + v for x in L}
    R = {Fraction(0)}
    for v in right:
        R = R | {x + v for x in R}
    if len(L) <= len(R):
        for l in L:
            if (target - l) in R:
                return True
    else:
        for r in R:
            if (target - r) in L:
                return True
    return False

def is_valid(S):
    S = list(set(S))
    if not S:
        return False
    r = {v: Fraction(1, v) for v in S}
    for a in S:
        others = [r[v] for v in S if v != a]
        if _subset_sum_possible(others, r[a]):
            return False
    return True

def construct():
    global _START
    _START = time.time()
    
    all_elems = list(range(1, 25))
    
    # We know that only the min element constraint matters.
    # So we can optimize the check.
    # But is_valid is robust.
    
    # Try to find a valid set of size 24, 23, 22, ...
    # Start with all elements, remove k elements.
    
    # To speed up, we can use the insight that only min(S) matters.
    # But let's stick to the robust is_valid for safety, 
    # and use the insight to guide the search order.
    
    # Search order:
    # 1. Try S = {1..24}
    # 2. Try S = {1..24} \ {x} for x in 2..24
    # 3. Try S = {1..24} \ {x, y}
    
    # We want to maximize size.
    # So we iterate k = 0, 1, 2, ...
    # For each k, we try combinations of k elements to remove from {2..24}.
    # We prioritize removing small elements.
    
    remove_candidates = list(range(2, 25))
    
    for k in range(0, 24):
        if _deadline_hit():
            break
        
        # Generate combinations of k elements to remove
        # To prioritize small elements, we can sort candidates and take first N?
        # No, we need to check ALL combinations to be sure, 
        # but we can order them so that "likely to work" are checked first.
        # Removing small numbers is likely to break sums.
        
        # Optimization: If we find a valid set, we return it immediately 
        # because we are iterating by increasing k (decreasing size).
        
        for rem in combinations(remove_candidates, k):
            if _deadline_hit():
                break
            S = [v for v in all_elems if v not in rem]
            if is_valid(S):
                return S
                
    # Fallback
    return list(range(13, 25))

if __name__ == "__main__":
    print(construct())
