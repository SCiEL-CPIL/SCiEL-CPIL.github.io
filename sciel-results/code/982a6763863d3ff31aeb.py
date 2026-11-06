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
    import time
    
    start_time = time.time()
    time_limit = 55  # Leave buffer for evaluation overhead
    
    def is_dissociated_fast(S):
        """Fast dissociation check with early termination."""
        if not S:
            return True
        L = lcm_list(S)
        weights = [L // n for n in S]
        k = len(weights)
        # Use iterative subset sum generation with early exit
        sums = {0}
        for w in weights:
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums = sums | new_sums
        return True
    
    # Try to find maximum cardinality from high to low
    best_result = None
    
    # Try sizes from 12 down to 1
    for target_size in range(12, 0, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try random search first
        random.seed(42)
        found = False
        max_trials = 20000 if target_size >= 10 else 50000
        
        for trial in range(max_trials):
            if time.time() - start_time > time_limit:
                break
            S = random.sample(range(1, 23), target_size)
            if is_dissociated_fast(S):
                found = True
                best_result = S
                break
        
        if found:
            break
        
        # If not found by random, try greedy for this size
        if target_size <= 10 and time.time() - start_time < time_limit:
            # Try greedy with multiple orderings
            orderings = [
                list(range(1, 23)),
                list(range(22, 0, -1)),
                list(range(1, 23, 2)) + list(range(2, 23, 2)),
                list(range(2, 23, 2)) + list(range(1, 23, 2)),
                [1, 3, 5, 7, 11, 13, 17, 19, 23, 2, 4, 6, 8, 10, 12, 14, 15, 16, 18, 20, 21, 22],
            ]
            for order in orderings:
                if time.time() - start_time > time_limit:
                    break
                S = []
                for n in order:
                    test_S = S + [n]
                    if len(test_S) <= target_size and is_dissociated_fast(test_S):
                        S = test_S
                if len(S) == target_size:
                    found = True
                    best_result = S
                    break
            if found:
                break
    
    # If nothing found, use fallback
    if best_result is None:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return sorted(best_result)


# EVOLVE-BLOCK-END
