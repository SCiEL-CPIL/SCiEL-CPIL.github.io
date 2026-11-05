import time
import math

def construct():
    """
    Constructs a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.
    
    For n = 255 = 3 * 5 * 17, the largest pairwise non-coprime set
    containing 255 is the set of all multiples of the smallest prime factor, 3.
    Any two distinct multiples of 3 have a gcd of at least 3.
    Size of this set is floor(255 / 3) = 85.
    
    We also explore whether removing small multiples of 3 and adding
    multiples of 5 or 17 can yield a larger set.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    n = 255
    # Base set: all multiples of 3 in [1, 255]
    base = list(range(3, n + 1, 3))
    best_set = set(base)
    best_size = len(base)
    
    # Candidates: numbers in [1,255] divisible by 5 or 17 but NOT by 3
    candidates = []
    for x in range(1, n + 1):
        if x % 3 != 0 and (x % 5 == 0 or x % 17 == 0):
            candidates.append(x)
    
    # Strategy: try removing small multiples of 3 to accommodate candidates
    # Element 3 conflicts with ALL non-multiples of 3, so removing it is key
    for remove_k in range(1, 15):
        if time.time() > deadline - 10:
            break
        # Remove the first remove_k multiples of 3: 3, 6, 9, ..., 3*remove_k
        to_remove = set(range(3, 3 * (remove_k + 1), 3))
        test_set = set(base) - to_remove
        
        # Greedily add candidates that don't conflict with test_set
        for x in sorted(candidates, key=lambda v: -v):  # Try larger first
            if x in test_set:
                continue
            # Check conflicts
            has_conflict = False
            for y in test_set:
                if math.gcd(x, y) == 1:
                    has_conflict = True
                    break
            if not has_conflict:
                test_set.add(x)
        
        if len(test_set) > best_size:
            best_set = test_set
            best_size = len(test_set)
    
    result = sorted(best_set)
    
    # Ensure 255 is in result
    if 255 not in result:
        result.append(255)
        result = sorted(result)
    
    # Verify pairwise non-coprime
    if time.time() < deadline - 5:
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback to base
            result = base
    
    return result
