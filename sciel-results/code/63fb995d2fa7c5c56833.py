"""Rewritten program for the reciprocal-subsum-free set task.

Goal: Find a subset S of [1, 24] that maximizes |S| such that for every
x in S, 1/x != sum of 1/y for any subset of S \ {x}.

Strategy:
  1. Exact validation using Fraction and meet-in-the-middle subset-sum.
  2. Heuristic search: greedy construction from various seeds, local
     improvement (add/remove/swap), and randomized restarts.
  3. Wall-clock deadline guard (75 s under the 90 s kill).
"""
import time
import random
from fractions import Fraction
from itertools import combinations

DEADLINE = 75.0
_START = None

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _subset_sum_possible(vals, target):
    """Check if target can be represented as sum of a subset of vals."""
    if not vals:
        return target == 0
    if len(vals) <= 16:
        # Brute force all subsets
        from itertools import combinations as _c
        s = set([Fraction(0)])
        for v in vals:
            s = s | {x + v for x in s}
        return target in s
    else:
        # Meet-in-the-middle
        n = len(vals)
        mid = n // 2
        left = vals[:mid]
        right = vals[mid:]
        
        # Generate all subset sums of left
        left_sums = set()
        for i in range(len(left)):
            for mask in range(1 << (i + 1)):
                s = Fraction(0)
                for j in range(i + 1):
                    if mask & (1 << j):
                        s += left[j]
                left_sums.add(s)
        # This is too slow. Let me do a better meet-in-the-middle.
        
        # Better approach:
        left_sums = set()
        for mask in range(1 << len(left)):
            s = Fraction(0)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
            
        right_sums = set()
        for mask in range(1 << len(right)):
            s = Fraction(0)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            right_sums.add(s)
            
        # Check if any left_sum + right_sum == target
        for ls in left_sums:
            if target - ls in right_sums:
                return True
        return False


def is_valid(S):
    """Check if S is reciprocal-subsum-free."""
    S_set = set(S)
    for x in S:
        target = Fraction(1, x)
        others = [Fraction(1, y) for y in S if y != x]
        if _subset_sum_possible(others, target):
            return False
    return True


def _greedy_build(order):
    """Greedy add elements in the given order."""
    S = []
    S_set = set()
    for x in order:
        if x in S_set:
            continue
        S_set.add(x)
        S.append(x)
        # Check validity
        if not is_valid(S):
            S_set.remove(x)
            S.pop()
    return S


def _greedy_remove(S):
    """Try to remove elements that cause conflicts."""
    S = list(S)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            # Try removing S[i]
            trial = S[:i] + S[i+1:]
            if len(trial) < len(S):
                # Check if removing improves validity (should already be valid if we started valid)
                pass
        # Actually, let's just return S if it's valid
    return S


def construct():
    global _START
    _START = time.time()
    
    best = list(range(13, 25))  # seed solution, size 12
    best_len = len(best)
    
    random.seed(42)
    
    # Strategy 1: Greedy construction from various orderings
    base_order = list(range(24, 0, -1))  # largest first
    S1 = _greedy_build(base_order)
    if len(S1) > best_len:
        best = S1
        best_len = len(S1)
    
    base_order2 = list(range(1, 25))  # smallest first
    S2 = _greedy_build(base_order2)
    if len(S2) > best_len:
        best = S2
        best_len = len(S2)
    
    # Strategy 2: Randomized greedy with multiple restarts
    for restart in range(50):
        if _deadline_hit():
            break
        
        order = list(range(1, 25))
        random.shuffle(order)
        S = _greedy_build(order)
        if len(S) > best_len:
            best = S
            best_len = len(S)
        
        # Local improvement: try adding elements
        improved = True
        while improved:
            if _deadline_hit():
                break
            improved = False
            S_set = set(S)
            for x in range(1, 25):
                if x in S_set:
                    continue
                trial = S + [x]
                if is_valid(trial):
                    S = trial
                    S_set.add(x)
                    improved = True
                    break
        
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 3: Try to extend the seed solution
    S = list(range(13, 25))
    S_set = set(S)
    for x in range(1, 13):
        if _deadline_hit():
            break
        trial = S + [x]
        if is_valid(trial):
            S = trial
            S_set.add(x)
    
    if len(S) > best_len:
        best = S
        best_len = len(S)
    
    # Strategy 4: Randomized local search from best
    for _ in range(20):
        if _deadline_hit():
            break
        
        S = list(best)
        S_set = set(S)
        
        # Try random swaps
        for _ in range(100):
            if _deadline_hit():
                break
            
            # Remove a random element
            i = random.randint(0, len(S) - 1)
            removed = S[i]
            S_new = S[:i] + S[i+1:]
            S_new_set = set(S_new)
            
            # Try adding a different element
            candidates = [x for x in range(1, 25) if x not in S_new_set]
            if not candidates:
                continue
            
            x = random.choice(candidates)
            trial = S_new + [x]
            if is_valid(trial) and len(trial) >= len(S):
                S = trial
                S_set = set(S)
        
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    return best
