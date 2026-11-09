"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _subset_sum_reaches(weights, target):
    """Check if any non-empty subset of weights sums to target."""
    if target <= 0:
        return False
    weights = [w for w in weights if 0 < w <= target]
    if not weights:
        return False
    weights.sort(reverse=True)
    achievable = {0}
    for w in weights:
        new_sums = set()
        for s in achievable:
            ns = s + w
            if ns == target:
                return True
            if ns < target:
                new_sums.add(ns)
        if new_sums:
            achievable |= new_sums
    return False

def _is_valid_set(S):
    """Check if S is reciprocal-subsum-free."""
    n = len(S)
    if n <= 1:
        return True
    L = _lcm_list(S)
    for i in range(n):
        x = S[i]
        target = L // x
        weights = [L // S[j] for j in range(n) if j != i]
        if _subset_sum_reaches(weights, target):
            return False
    return True

def _can_add(S, x):
    """Check if x can be added to S while maintaining validity."""
    if x in S:
        return False
    return _is_valid_set(S + [x])

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best = list(range(13, 25))
    best_size = len(best)
    
    # Greedily add smaller numbers
    for x in range(12, 0, -1):
        if time.time() > deadline:
            break
        if _can_add(best, x):
            best.append(x)
            best_size = len(best)
    
    # Local search
    random.seed(42)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_set = set(best)
        
        # Try adding each missing element
        for x in range(1, 25):
            if time.time() > deadline:
                break
            if x not in S_set:
                if _can_add(best, x):
                    best.append(x)
                    S_set.add(x)
                    best_size = len(best)
                    improved = True
        
        # Try 1-for-2 swaps: remove one element, add two
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for x in range(1, 25):
                if time.time() > deadline:
                    break
                if x not in S_set:
                    for y in range(1, 25):
                        if time.time() > deadline:
                            break
                        if y not in S_set and y != x:
                            new_S = best[:i] + best[i+1:] + [x, y]
                            if len(new_S) > len(best) and _is_valid_set(new_S):
                                best = new_S
                                best_size = len(best)
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
    
    return best
# EVOLVE-BLOCK-END
