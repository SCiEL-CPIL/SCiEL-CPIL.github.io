import time
from fractions import Fraction

def _is_valid(S):
    """Check if S satisfies the reciprocal-subsum-free condition."""
    S_set = set(S)
    for x in S_set:
        target = Fraction(1, x)
        others = [y for y in S_set if y != x]
        # Check all non-empty subsets of others
        n = len(others)
        for mask in range(1, 1 << n):
            total = Fraction(0, 1)
            for i in range(n):
                if mask & (1 << i):
                    total += Fraction(1, others[i])
            if total == target:
                return False
    return True

def _greedy_large_to_small(deadline):
    """Greedy: add largest first, check if 1/x is subset sum of current."""
    current = []
    sums = {Fraction(0, 1)}
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
        target = Fraction(1, x)
        if target in sums:
            continue
        current.append(x)
        new_sums = set()
        for s in sums:
            ns = s + target
            if ns <= 1:
                new_sums.add(ns)
        sums |= new_sums
    return current if current else [24]

def _greedy_small_to_large(deadline):
    """Greedy: add smallest first, check if 1/x is subset sum of current."""
    current = []
    sums = {Fraction(0, 1)}
    for x in range(1, 25):
        if time.time() > deadline:
            break
        target = Fraction(1, x)
        if target in sums:
            continue
        current.append(x)
        new_sums = set()
        for s in sums:
            ns = s + target
            if ns <= 1:
                new_sums.add(ns)
        sums |= new_sums
    return current if current else [24]

def _greedy_with_order(order, deadline):
    """Greedy with a custom order."""
    current = []
    sums = {Fraction(0, 1)}
    for x in order:
        if time.time() > deadline:
            break
        target = Fraction(1, x)
        if target in sums:
            continue
        current.append(x)
        new_sums = set()
        for s in sums:
            ns = s + target
            if ns <= 1:
                new_sums.add(ns)
        sums |= new_sums
    return current if current else [24]

def _local_search(S, deadline):
    """Try to improve S by removing elements or swapping."""
    S_set = set(S)
    best = list(S_set)
    best_size = len(best)
    
    # Try removing each element and see if we can add others
    for x in list(S_set):
        if time.time() > deadline:
            break
        candidate = S_set - {x}
        # Try to add back skipped elements
        skipped = set(range(1, 25)) - candidate
        for y in sorted(skipped, reverse=True):
            if time.time() > deadline:
                break
            test = candidate | {y}
            if _is_valid(test):
                candidate = test
        if len(candidate) > best_size:
            best = list(candidate)
            best_size = len(best)
    
    return best

def construct():
    start = time.time()
    deadline = start + 80.0
    
    # Try multiple greedy orders
    candidates = []
    
    # Standard large-to-small
    c1 = _greedy_large_to_small(deadline)
    candidates.append(c1)
    
    # Small-to-large
    c2 = _greedy_small_to_large(deadline)
    candidates.append(c2)
    
    # Try some custom orders
    # Even numbers first, then odd
    order_even_odd = list(range(2, 25, 2)) + list(range(1, 24, 2))
    c3 = _greedy_with_order(order_even_odd, deadline)
    candidates.append(c3)
    
    # Odd first, then even
    order_odd_even = list(range(1, 25, 2)) + list(range(2, 25, 2))
    c4 = _greedy_with_order(order_odd_even, deadline)
    candidates.append(c4)
    
    # Primes first
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    non_primes = [x for x in range(1, 25) if x not in primes]
    order_primes = primes + non_primes
    c5 = _greedy_with_order(order_primes, deadline)
    candidates.append(c5)
    
    # Composite first
    composites = [x for x in range(4, 25) if x not in primes and x > 1]
    order_composites = composites + primes
    c6 = _greedy_with_order(order_composites, deadline)
    candidates.append(c6)
    
    # Pick the best candidate by size
    best = max(candidates, key=len)
    
    # Try local search to improve
    if time.time() < deadline:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if not _is_valid(best):
        # Fallback: try to fix by removing offending elements
        S_set = set(best)
        for x in list(S_set):
            test = S_set - {x}
            if _is_valid(test):
                S_set = test
        best = list(S_set)
        if not best:
            best = [24]
    
    return best
