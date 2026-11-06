import time
from fractions import Fraction
import random

_START_TIME = time.time()
DEADLINE = 85.0  # 5 seconds safety margin

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _check_violation(S, a):
    """
    Check if 1/a can be formed by a subset of {1/x for x in S, x != a}.
    Returns the subset T if found, else None.
    """
    target = Fraction(1, a)
    # Items are 1/x for x in S, x != a, and 1/x < target (i.e., x > a)
    # If 1/x > target, it cannot be part of the sum (since all terms are positive)
    items = []
    for x in S:
        if x != a:
            val = Fraction(1, x)
            if val < target:
                items.append((val, x))
            # If val == target, then {x} is a solution
            elif val == target:
                return [x]
    
    if not items:
        return None
        
    # Sort by value descending to prune faster
    items.sort(key=lambda p: p[0], reverse=True)
    values = [p[0] for p in items]
    origs = [p[1] for p in items]
    
    # Suffix sums for pruning: sum of items from i to end
    n = len(items)
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + values[i]
        
    def dfs(idx, current_sum, current_set):
        if current_sum == target:
            return current_set
        if idx >= n:
            return None
        if current_sum > target:
            return None
        if current_sum + suffix_sums[idx] < target:
            return None
            
        # Option 1: Include items[idx]
        # Only if it doesn't exceed target
        if current_sum + values[idx] <= target:
            res = dfs(idx + 1, current_sum + values[idx], current_set + [origs[idx]])
            if res is not None:
                return res
        
        # Option 2: Exclude items[idx]
        res = dfs(idx + 1, current_sum, current_set)
        if res is not None:
            return res
            
        return None

    return dfs(0, Fraction(0), [])

def is_valid(S):
    for a in S:
        if _check_violation(S, a) is not None:
            return False
    return True

def get_violation(S):
    """Returns (a, T) if violation exists, else None."""
    for a in S:
        T = _check_violation(S, a)
        if T is not None:
            return (a, T)
    return None

def greedy_solve(strategy='min'):
    S = list(range(1, 25))
    while _time_left() > 1.0:
        viol = get_violation(S)
        if viol is None:
            return S
        a, T = viol
        candidates = [a] + T
        if strategy == 'min':
            y = min(candidates)
        elif strategy == 'max':
            y = max(candidates)
        elif strategy == 'random':
            y = random.choice(candidates)
        else:
            y = min(candidates)
        S.remove(y)
    return S

def construct():
    best_set = None
    best_size = 0
    
    strategies = ['min', 'max', 'random']
    
    for strat in strategies:
        if _time_left() < 2.0:
            break
        S = greedy_solve(strat)
        if S and is_valid(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
                
    # If we have time, try random restarts or perturbations?
    # The greedy 'min' usually gives 19.
    # Let's try to improve by removing fewer elements if possible.
    # Actually, the greedy process is deterministic for min/max.
    # For random, it varies.
    
    if best_set is None:
        # Fallback
        best_set = list(range(13, 25))
        
    return best_set
