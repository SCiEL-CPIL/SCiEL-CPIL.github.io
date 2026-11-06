"""Rewritten program for the reciprocal-subsum-free set task."""
import time
from fractions import Fraction
from itertools import combinations

_START = None
DEADLINE = 55.0 # Safety margin

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _subset_sum_check(target, candidates):
    """
    Check if 'target' (Fraction) can be formed by a subset sum of reciprocals of 'candidates'.
    'candidates' is a list of integers.
    Uses meet-in-the-middle for efficiency if len(candidates) > 10, else brute force.
    """
    if target <= 0:
        return False
    if not candidates:
        return False
    
    n = len(candidates)
    
    # If target is very small, or candidates are large, pruning might help, 
    # but for n <= 20, MITM is robust.
    
    if n <= 15:
        # Brute force with pruning
        # Sort candidates by value (smaller value -> larger reciprocal)
        # Actually, just iterate masks
        for mask in range(1, 1 << n):
            s = Fraction(0, 1)
            # Early termination if sum exceeds target?
            # Reciprocals are positive.
            for i in range(n):
                if mask & (1 << i):
                    s += Fraction(1, candidates[i])
                    if s > target:
                        break
            else:
                if s == target:
                    return True
        return False
    else:
        # Meet in the middle
        half = n // 2
        left = candidates[:half]
        right = candidates[half:]
        
        left_sums = set()
        # Generate all subset sums for left
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            left_sums.add(s)
            
        # Check right subsets
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            if (target - s) in left_sums:
                return True
        return False

def _is_valid_addition(S, v):
    """
    Check if adding v to S (where all u in S < v) maintains validity.
    S is a list of selected integers.
    """
    for u in S:
        # We need to check if 1/u - 1/v is a subset sum of {w in S : w > u}
        target = Fraction(1, u) - Fraction(1, v)
        if target <= 0:
            continue # Should not happen since u < v => 1/u > 1/v
        
        # Candidates are elements in S that are greater than u
        # Since S is sorted, we can take elements after u
        # Note: u is in S. We need w in S such that w > u.
        # Also w != v (v is not in S yet).
        # The pool is {w in S | w > u}.
        
        # Optimization: If the sum of all reciprocals in the pool is less than target, it's impossible.
        # Also, if the smallest reciprocal in the pool is greater than target, it's impossible (if only one term allowed? No, sum of multiple).
        
        pool = [w for w in S if w > u]
        
        if not pool:
            continue
            
        # Quick check: sum of all reciprocals in pool
        total_sum = sum(Fraction(1, w) for w in pool)
        if total_sum < target:
            continue
            
        # Check subset sum
        if _subset_sum_check(target, pool):
            return False
            
    return True

def _solve():
    """
    Backtracking search to find the maximum size valid subset of [1, 24].
    """
    candidates = list(range(1, 25))
    best_set = []
    best_size = 0
    
    # We process candidates in increasing order.
    # State: index in candidates, current set S
    def backtrack(idx, S):
        nonlocal best_set, best_size
        
        if _deadline_hit():
            return
            
        # Pruning: even if we take all remaining, can we beat best?
        if len(S) + (len(candidates) - idx) <= best_size:
            return
            
        if idx == len(candidates):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
            
        # Option 1: Exclude candidates[idx]
        backtrack(idx + 1, S)
        
        # Option 2: Include candidates[idx]
        v = candidates[idx]
        
        # Check if adding v is valid
        if _is_valid_addition(S, v):
            S.append(v)
            backtrack(idx + 1, S)
            S.pop()
            
    backtrack(0, [])
    
    return best_set

def construct():
    global _START
    _START = time.time()
    
    result = _solve()
    
    # Safety: ensure nonempty
    if not result:
        result = [24]
        
    return result
