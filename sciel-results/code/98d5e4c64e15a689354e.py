"""Rewritten program: exact LCM-based integer subset-sum validation with greedy + local search."""
import time
import math
from itertools import combinations
from functools import lru_cache

# Deadline guard
_DEADLINE = 85.0  # seconds (90s limit, 5s safety margin)
_start_time = None

def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time

def _check_deadline():
    return _elapsed() > _DEADLINE

def construct():
    global _start_time
    _start_time = time.time()
    
    # We want a nonempty subset of [1, 24] that is "reciprocal-subsum-free":
    # For no selected element x, 1/x == sum of 1/y for a subset of the OTHER selected elements.
    
    # Strategy: 
    # 1. Precompute LCM of 1..24 to convert reciprocal sums to integers.
    # 2. Use greedy construction with multiple orderings, then local search (add/remove/swap) to maximize size.
    
    # LCM of 1..24
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
    # L is LCM(1..24) = 5354228880
    
    # For each element x in [1,24], 1/x = L/x (as integer)
    # A subset S of elements is valid if for every x in S:
    #   L/x is NOT equal to sum of L/y for any nonempty subset T of (S \ {x})
    #   Note: the empty subset sums to 0, and L/x > 0, so empty subset never matches.
    
    # Precompute: for each x, target = L // x
    targets = {x: L // x for x in range(1, 25)}
    
    # For a given set S (as frozenset), check validity:
    def is_valid(S):
        S_list = list(S)
        n = len(S_list)
        if n == 0:
            return True
        # For each x in S, check if target[x] is achievable as sum of L/y for y in S\{x}
        # We need to check: does there exist a nonempty subset T of S\{x} such that sum(L/y for y in T) == target[x]?
        # Since L/y are positive integers, we can use subset sum.
        # But doing this naively is expensive. Let's precompute for each candidate set.
        
        for x in S_list:
            others = [y for y in S_list if y != x]
            if not others:
                continue
            target_val = targets[x]
            # Check if target_val can be formed as sum of a subset of {L//y for y in others}
            # Use a set-based subset sum (since values can be large, use set of achievable sums)
            # But the number of others can be up to 23, so 2^23 is too many.
            # Instead, use a boolean DP or a set of achievable sums, pruning large sums.
            
            # Actually, since we're dealing with specific values, let's use a set of achievable sums.
            # The maximum possible sum is sum(L//y for y in others), which for small y can be large.
            # But we only care if target_val is achievable. We can prune sums > target_val.
            
            achievable = {0}
            for y in others:
                val = L // y
                if val > target_val:
                    continue
                new_sums = set()
                for s in achievable:
                    ns = s + val
                    if ns == target_val:
                        return False  # Found a subset summing to target_val
                    if ns < target_val:
                        new_sums.add(ns)
                achievable |= new_sums
                if target_val in achievable:
                    return False
            # If we get here, x is safe
        return True
    
    # Greedy construction: try adding elements in various orders
    def greedy_construct(order):
        S = set()
        for x in order:
            if _check_deadline():
                break
            candidate = S | {x}
            if is_valid(candidate):
                S = candidate
        return S
    
    # Try multiple orderings
    best_set = set()
    
    # Order 1: descending (larger elements first, smaller reciprocals)
    order_desc = list(range(24, 0, -1))
    S1 = greedy_construct(order_desc)
    if len(S1) > len(best_set):
        best_set = S1
    
    if _check_deadline():
        return list(best_set) if best_set else [24]
    
    # Order 2: ascending
    order_asc = list(range(1, 25))
    S2 = greedy_construct(order_asc)
    if len(S2) > len(best_set):
        best_set = S2
    
    if _check_deadline():
        return list(best_set) if best_set else [24]
    
    # Order 3: even numbers first, then odd
    order_even_odd = list(range(2, 25, 2)) + list(range(1, 24, 2))
    S3 = greedy_construct(order_even_odd)
    if len(S3) > len(best_set):
        best_set = S3
    
    if _check_deadline():
        return list(best_set) if best_set else [24]
    
    # Order 4: odd numbers first, then even
    order_odd_even = list(range(1, 25, 2)) + list(range(2, 24, 2))
    S4 = greedy_construct(order_odd_even)
    if len(S4) > len(best_set):
        best_set = S4
    
    if _check_deadline():
        return list(best_set) if best_set else [24]
    
    # Local search: try to add elements not in best_set, or swap
    improved = True
    while improved and not _check_deadline():
        improved = False
        # Try adding each element not in best_set
        for x in range(1, 25):
            if x in best_set:
                continue
            candidate = best_set | {x}
            if is_valid(candidate):
                best_set = candidate
                improved = True
                break
        if _check_deadline():
            break
        # Try swapping: remove one, add another
        for x in list(best_set):
            if _check_deadline():
                break
            for y in range(1, 25):
                if y in best_set or y == x:
                    continue
                candidate = (best_set - {x}) | {y}
                if is_valid(candidate):
                    if len(candidate) > len(best_set):
                        best_set = candidate
                        improved = True
                        break
        if _check_deadline():
            break
    
    # Final: also try a more systematic local search with 2-swaps
    if not _check_deadline():
        improved = True
        while improved and not _check_deadline():
            improved = False
            for x1 in list(best_set):
                if _check_deadline():
                    break
                for x2 in list(best_set):
                    if x2 <= x1:
                        continue
                    if _check_deadline():
                        break
                    for y1 in range(1, 25):
                        if _check_deadline():
                            break
                        if y1 in best_set or y1 == x1 or y1 == x2:
                            continue
                        for y2 in range(y1 + 1, 25):
                            if _check_deadline():
                                break
                            if y2 in best_set or y2 == x1 or y2 == x2:
                                continue
                            candidate = (best_set - {x1, x2}) | {y1, y2}
                            if is_valid(candidate) and len(candidate) > len(best_set):
                                best_set = candidate
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    if not best_set:
        return [24]
    
    return list(best_set)
