"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
from functools import reduce
from itertools import combinations

def lcm(a, b):
    return a * b // gcd(a, b)

def lcm_list(lst):
    return reduce(lcm, lst, 1)

def is_dissociated(S):
    """Check if all subset sums of reciprocals of S are distinct, using integer scaling."""
    if not S:
        return True
    L = lcm_list(S)
    # Scaled weights
    weights = [L // n for n in S]
    k = len(weights)
    # Generate all 2^k subset sums
    subset_sums = set()
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += weights[i]
        if s in subset_sums:
            return False
        subset_sums.add(s)
    return True

def construct():
    import random
    
    candidates = list(range(1, 23))
    best_cardinality = 0
    best_result = None
    
    # Try greedy with many random orderings
    # Use deterministic seeds for reproducibility
    for i in range(2000):
        random.seed(i)
        order = candidates[:]
        random.shuffle(order)
        
        S = []
        for n in order:
            test_S = S + [n]
            if is_dissociated(test_S):
                S = test_S
        
        if len(S) > best_cardinality:
            best_cardinality = len(S)
            best_result = S
            
        # If we found a set of size 12 or more, it's likely near optimal
        if best_cardinality >= 12:
            break

    # Try to extend the best set found by adding any remaining element
    if best_result is not None:
        S = list(best_result)
        # Perform a few passes of extension
        for _ in range(10):
            changed = False
            # Shuffle candidates to avoid bias
            random.seed(best_cardinality + 100)
            cand_copy = candidates[:]
            random.shuffle(cand_copy)
            for n in cand_copy:
                if n not in S:
                    test_S = S + [n]
                    if is_dissociated(test_S):
                        S = test_S
                        changed = True
                        break
            if not changed:
                break
        
        if len(S) > best_cardinality:
            best_cardinality = len(S)
            best_result = S

    # Fallback to known good set if nothing found (should not happen)
    if best_result is None:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return sorted(best_result)

# EVOLVE-BLOCK-END
