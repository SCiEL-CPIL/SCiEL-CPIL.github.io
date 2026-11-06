"""Program for large reciprocal-subsum-free sets using greedy construction with backtracking."""
import time
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each x in S, 1/x must NOT equal sum(1/y for y in T)
    where T is a non-empty subset of S \\ {x}.
    """
    if not S:
        return True
    
    S_list = sorted(S, reverse=True)
    n = len(S_list)
    
    # Precompute reciprocals
    recip = {x: Fraction(1, x) for x in S_list}
    
    for i, x in enumerate(S_list):
        # Check if 1/x equals sum of reciprocals of any non-empty subset of others
        target = recip[x]
        others = [y for j, y in enumerate(S_list) if j != i]
        
        # Subset sum check: can we form 'target' from reciprocals of 'others'?
        # Use DFS with pruning
        others_sorted = sorted(others, key=lambda y: recip[y], reverse=True)
        others_recips = [recip[y] for y in others_sorted]
        m = len(others_recips)
        
        # If sum of all others < target, impossible
        if sum(others_recips) < target:
            continue
        
        # DFS to find subset summing to target
        def dfs(idx, current):
            if current == target:
                return True
            if idx >= m or current > target:
                return False
            # Include
            if dfs(idx + 1, current + others_recips[idx]):
                return True
            # Exclude
            if dfs(idx + 1, current):
                return True
            return False
        
        # Optimization: try to find a quick contradiction
        # If target is less than the smallest reciprocal, impossible
        if target < others_recips[-1]:
            continue
            
        if dfs(0, Fraction(0)):
            return False
    
    return True


def _can_add(k, selected):
    """Check if adding k to selected keeps the set subsum-free."""
    new_set = set(selected) | {k}
    return _is_subsum_free(new_set)


def construct():
    """
    Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximizing cardinality.
    """
    best_set = set(range(13, 25))  # Fallback: the seed answer
    best_size = len(best_set)
    
    # Greedy descending: try to add elements from 24 down to 1
    # This tends to work well because large numbers have small reciprocals
    # and are less likely to form subsums
    
    current = set()
    candidates = list(range(24, 0, -1))
    
    for k in candidates:
        if _time_remaining() < 5.0:
            break
        if _can_add(k, current):
            current.add(k)
    
    if len(current) > best_size:
        best_set = current
        best_size = len(current)
    
    # Try to improve with local search / backtracking
    # Try adding elements we skipped, and removing others to make room
    # Use a simple backtracking approach with time limit
    
    def backtrack(selected, remaining, depth=0):
        nonlocal best_set, best_size
        
        if _time_remaining() < 2.0:
            return
        
        if len(selected) > best_size:
            best_set = set(selected)
            best_size = len(selected)
        
        if not remaining:
            return
        
        # Try to add the next candidate
        for i, k in enumerate(remaining):
            if _time_remaining() < 2.0:
                return
            
            # Try including k
            if _can_add(k, selected):
                new_selected = set(selected) | {k}
                backtrack(new_selected, remaining[i+1:], depth+1)
            
            # Try excluding k (just continue without it)
            # But this is implicit in trying the next candidate
        
        # Also try to see if we can add any element from the full range
        # that's not in selected or remaining
        all_candidates = set(range(1, 25))
        for k in sorted(all_candidates - selected, reverse=True):
            if _time_remaining() < 2.0:
                return
            if k not in remaining:  # Already considered in this branch
                continue
            if _can_add(k, selected):
                new_selected = set(selected) | {k}
                backtrack(new_selected, remaining, depth+1)
    
    # Start backtracking from the greedy solution
    remaining = list(range(24, 0, -1))
    backtrack(set(current), remaining)
    
    # Final check: ensure we return a valid nonempty set
    if not best_set:
        best_set = {24}
    
    return list(best_set)
