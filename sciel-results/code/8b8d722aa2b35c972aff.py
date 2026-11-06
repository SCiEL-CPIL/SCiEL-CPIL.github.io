"""Combined program for reciprocal-subsum-free set task."""
import time
from fractions import Fraction

_START = None
DEADLINE = 85.0  # Safety margin before 90s kill

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _subset_sum_check(target, candidates):
    """
    Check if 'target' (Fraction) can be formed by a subset sum of reciprocals of 'candidates'.
    Uses meet-in-the-middle for efficiency.
    """
    if target <= 0 or not candidates:
        return False
    
    n = len(candidates)
    
    if n <= 12:
        # Brute force with early termination
        for mask in range(1, 1 << n):
            s = Fraction(0, 1)
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
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            left_sums.add(s)
            
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            if (target - s) in left_sums:
                return True
        return False

def _check_violation(S, a):
    """Check if 1/a equals sum of reciprocals of some subset of S \\ {a}."""
    others = [x for x in S if x != a]
    if not others:
        return False
    
    target = Fraction(1, a)
    n = len(others)
    
    if n <= 12:
        for mask in range(1, 1 << n):
            s = Fraction(0, 1)
            for i in range(n):
                if mask & (1 << i):
                    s += Fraction(1, others[i])
                    if s > target:
                        break
            else:
                if s == target:
                    return True
        return False
    else:
        half = n // 2
        left = others[:half]
        right = others[half:]
        
        left_sums = set()
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            left_sums.add(s)
            
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            if (target - s) in left_sums:
                return True
        return False

def _is_valid(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    for a in S:
        if _check_violation(S, a):
            return False
    return True

def _is_valid_addition(S, v):
    """
    Check if adding v to S maintains validity.
    S is a list of selected integers (sorted).
    """
    for u in S:
        target = Fraction(1, u) - Fraction(1, v)
        if target <= 0:
            continue
        
        pool = [w for w in S if w > u]
        if not pool:
            continue
            
        total_sum = sum(Fraction(1, w) for w in pool)
        if total_sum < target:
            continue
            
        if _subset_sum_check(target, pool):
            return False
            
    return True

def _greedy_remove():
    """Strategy: Start with all [1,24], remove elements that cause violations."""
    S = list(range(1, 25))
    changed = True
    while changed and not _deadline_hit():
        changed = False
        for a in sorted(S):
            if _deadline_hit():
                return S
            if _check_violation(S, a):
                S.remove(a)
                changed = True
                break
    return S

def _greedy_add(order):
    """Strategy: Greedy add elements in given order."""
    S = []
    for a in order:
        if _deadline_hit():
            break
        if _is_valid_addition(S, a):
            S.append(a)
    return S

def _local_search(S):
    """Try to improve S by swapping or adding/removing elements."""
    S = list(S)
    improved = True
    while improved and not _deadline_hit():
        improved = False
        
        # Try adding each missing element
        missing = [x for x in range(1, 25) if x not in S]
        for a in missing:
            if _deadline_hit():
                return S
            new_S = S + [a]
            if _is_valid(new_S):
                S = new_S
                improved = True
                break
        
        # Try removing and re-adding to find better combinations
        for i in range(len(S)):
            if _deadline_hit():
                return S
            S_copy = S[:]
            removed = S_copy.pop(i)
            # Try adding other missing elements
            missing = [x for x in range(1, 25) if x not in S_copy]
            for a in missing:
                if _deadline_hit():
                    return S
                if a == removed:
                    continue
                new_S = S_copy + [a]
                if _is_valid(new_S) and len(new_S) >= len(S):
                    S = new_S
                    improved = True
                    break
            if improved:
                break
    return S

def _backtracking_search():
    """Backtracking search to find large valid sets."""
    candidates = list(range(1, 25))
    best_set = []
    best_size = 0
    
    def backtrack(idx, S):
        nonlocal best_set, best_size
        
        if _deadline_hit():
            return
            
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
        if _is_valid_addition(S, v):
            S.append(v)
            backtrack(idx + 1, S)
            S.pop()
            
    backtrack(0, [])
    return best_set

def construct():
    global _START
    _START = time.time()
    
    # Start with the known good seed
    best = list(range(13, 25))  # 12 elements
    
    # Strategy 1: Greedy removal from full set
    S1 = _greedy_remove()
    if len(S1) > len(best):
        best = S1
    
    # Strategy 2: Greedy addition in various orders
    orders = [
        range(24, 0, -1),  # descending
        range(1, 25),      # ascending
        range(12, 0, -1),  # lower half descending
        range(24, 12, -1), # upper half descending
        [24, 1, 23, 2, 22, 3, 21, 4, 20, 5, 19, 6, 18, 7, 17, 8, 16, 9, 15, 10, 14, 11, 13, 12]
    ]
    for order in orders:
        if _deadline_hit():
            break
        S2 = _greedy_add(list(order))
        if len(S2) > len(best):
            best = S2
    
    # Strategy 3: Local search from best found so far
    S3 = _local_search(best)
    if len(S3) > len(best):
        best = S3
    
    # Strategy 4: Backtracking (use remaining time)
    if not _deadline_hit():
        S4 = _backtracking_search()
        if len(S4) > len(best):
            best = S4
    
    # Final safety: ensure nonempty
    if not best:
        best = [24]
        
    return best
