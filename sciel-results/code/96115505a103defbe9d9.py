import time
import random
from math import gcd
from functools import reduce

# Calculate LCM of 1..24
def lcm(a, b):
    return a * b // gcd(a, b)

LCM = reduce(lcm, range(1, 25))
# Map 1/y to integer value
VAL = {i: LCM // i for i in range(1, 25)}

def find_violation(S):
    """
    Returns (x, T) such that 1/x = sum(1/y for y in T) and T subset of S\{x}, or None.
    S is a set of integers.
    """
    S_list = sorted(S)
    for x in S_list:
        target = VAL[x]
        # Candidates must be > x
        candidates = [y for y in S_list if y > x]
        if not candidates:
            continue
        
        # Subset sum check with reconstruction
        # We want to find a subset of candidates that sums to target.
        # DFS
        n = len(candidates)
        # Sort candidates by value descending (larger 1/y first) to prune?
        # Actually, larger integers (smaller y) are larger values.
        # 1/y is larger for smaller y.
        # So candidates sorted by y ascending means values descending.
        
        def dfs(idx, current_sum, current_set):
            if current_sum == target:
                return current_set
            if idx == n:
                return None
            # Pruning: if current_sum > target, return None
            if current_sum > target:
                return None
            
            # Option 1: Skip candidates[idx]
            res = dfs(idx + 1, current_sum, current_set)
            if res is not None:
                return res
            
            # Option 2: Take candidates[idx]
            v = VAL[candidates[idx]]
            if current_sum + v <= target:
                res = dfs(idx + 1, current_sum + v, current_set + [candidates[idx]])
                if res is not None:
                    return res
            return None
        
        T = dfs(0, 0, [])
        if T is not None:
            return x, T
            
    return None

def repair(S, strategy='target', deadline=None):
    S = set(S)
    while True:
        if deadline and time.time() > deadline:
            break
        v = find_violation(S)
        if v is None:
            break
        x, T = v
        if strategy == 'target':
            S.remove(x)
        elif strategy == 'max_summand':
            S.remove(max(T))
        elif strategy == 'min_summand':
            S.remove(min(T))
        elif strategy == 'random':
            S.remove(random.choice([x] + T))
        else:
            S.remove(x)
    return S

def try_add(S):
    """
    Try to add elements from 1..24 not in S to S, maintaining validity.
    Returns the new set.
    """
    S = set(S)
    for z in range(1, 25):
        if z in S:
            continue
        # Check if S U {z} is valid
        new_S = S | {z}
        if find_violation(new_S) is None:
            S = new_S
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe
    
    best_set = set()
    
    strategies = ['target', 'max_summand', 'min_summand', 'random']
    
    # Run each strategy
    for strat in strategies:
        if time.time() > deadline:
            break
        S = repair(set(range(1, 25)), strategy=strat, deadline=deadline)
        S = try_add(S)
        if len(S) > len(best_set):
            best_set = S
            
    # If we have time, do some random restarts with 'random' strategy
    while time.time() < deadline - 5.0:
        S = repair(set(range(1, 25)), strategy='random', deadline=deadline)
        S = try_add(S)
        if len(S) > len(best_set):
            best_set = S
            
    # If best_set is empty (should not happen), fallback
    if not best_set:
        best_set = set(range(13, 25))
        
    return list(best_set)
