"""Improved program with multiple greedy strategies for the 'no element divides sum of 3 distinct others' task."""

import random

def _can_add(x, S):
    """Check if x can be added to set S without violating constraints."""
    nS = len(S)
    
    # 1. Check constraints imposed by x on the existing set S
    # No 3 distinct elements in S should sum to a multiple of x
    if nS >= 3:
        for i in range(nS):
            for j in range(i + 1, nS):
                for k in range(j + 1, nS):
                    s = S[i] + S[j] + S[k]
                    if s % x == 0:
                        return False
    
    # 2. Check constraints imposed by existing elements in S on the new triplets involving x
    # For each a in S, no pair (b, c) in S (b != a, c != a, b != c) such that (x + b + c) % a == 0
    for idx_a, a in enumerate(S):
        for i in range(nS):
            if i == idx_a:
                continue
            for j in range(i + 1, nS):
                if j == idx_a:
                    continue
                s = x + S[i] + S[j]
                if s % a == 0:
                    return False
    
    return True

def _greedy(order):
    """Run greedy construction with given order of candidates."""
    S = []
    for x in order:
        if _can_add(x, S):
            S.append(x)
    return S

def construct():
    """
    Returns a list of distinct positive integers <= 192, maximizing cardinality,
    such that no selected integer divides the sum of any 3 distinct OTHER selected integers.
    """
    n = 192
    candidates = list(range(1, n + 1))
    
    best = []
    
    # Strategy 1: Descending order (large numbers first)
    result = _greedy(sorted(candidates, reverse=True))
    if len(result) > len(best):
        best = result
    
    # Strategy 2: Ascending order
    result = _greedy(sorted(candidates))
    if len(result) > len(best):
        best = result
    
    # Strategy 3: Random permutations (try several)
    random.seed(42)
    for _ in range(50):
        perm = candidates[:]
        random.shuffle(perm)
        result = _greedy(perm)
        if len(result) > len(best):
            best = result
    
    # Strategy 4: Try to add back skipped numbers in a second pass
    # For each number not in best, try to add it if possible
    in_best = set(best)
    for x in range(n, 0, -1):
        if x not in in_best:
            S = best[:]
            if _can_add(x, S):
                S.append(x)
                if len(S) > len(best):
                    best = S
                    in_best = set(best)
    
    return best
