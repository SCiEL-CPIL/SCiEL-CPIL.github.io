import itertools
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def lcm_list(lst):
    return reduce(lcm, lst)

# Precompute LCM of 1..24
L = lcm_list(range(1, 25))

def is_valid_subset(S):
    """
    Check if S is reciprocal-subsum-free.
    S is a list of integers.
    """
    S_set = set(S)
    weights = {x: L // x for x in S_set}
    
    # For each a in S, check if weights[a] is a subset sum of weights of S \ {a}
    for a in S_set:
        target = weights[a]
        others = [x for x in S_set if x != a]
        # Check if target is a subset sum of weights of others
        # Use DP or bitset? 
        # Max sum is small enough for a set of sums?
        # Number of elements is ~20. 2^20 is 1M. Too slow for 20 checks in a loop if called many times.
        # But we only call it a few times.
        
        # Meet in the middle?
        # Or simple DFS with pruning.
        
        # Since we only need to check a few specific sets, we can use a simple recursive check.
        n = len(others)
        if n == 0:
            continue
            
        # Sort others by weight descending to prune earlier?
        # Actually, we just need to find ANY subset.
        
        # Use a set of reachable sums
        sums = {0}
        for b in others:
            w = weights[b]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns == target:
                    return False
                if ns < target:
                    new_sums.add(ns)
            sums.update(new_sums)
            
    return True

def construct():
    # Candidate set from manual analysis
    S = [1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    
    # Verify
    if not is_valid_subset(S):
        # Fallback to safe set
        return list(range(13, 25))
        
    # Try to add missing elements
    missing = [x for x in range(1, 25) if x not in S]
    # Sort missing by value? Or try all.
    # We want to maximize size.
    # Try adding each missing element.
    best_S = S[:]
    for m in missing:
        new_S = best_S + [m]
        if is_valid_subset(new_S):
            best_S = new_S
            
    # Try swapping? 
    # If we can't add, maybe we can swap a small element in best_S with a missing one?
    # But best_S is already large.
    
    return best_S
