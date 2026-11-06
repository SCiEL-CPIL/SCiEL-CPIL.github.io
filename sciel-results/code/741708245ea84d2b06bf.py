"""Rewritten program: greedy + local search for large reciprocal-subsum-free sets."""
import time
from itertools import combinations

DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    S = list(S)
    n = len(S)
    if n == 0:
        return True
    # Precompute reciprocals
    recips = [1.0 / x for x in S]
    # For each element i, check that 1/S[i] != sum of reciprocals of any subset of others
    # Use exact rational arithmetic to avoid floating point issues
    from fractions import Fraction
    frac = [Fraction(1, x) for x in S]
    for i in range(n):
        target = frac[i]
        others = [j for j in range(n) if j != i]
        m = len(others)
        # Enumerate all subsets of others (2^m subsets). For m up to ~18 this is 262144, fine.
        # But for larger m we need pruning. Since we're maximizing, m could be up to 24.
        # 2^24 = 16M which is too slow per check. Use meet-in-the-middle or smarter.
        # For now, if m <= 20, enumerate. Otherwise use a different approach.
        if m <= 20:
            for mask in range(1, 1 << m):
                s = Fraction(0, 1)
                for k in range(m):
                    if mask & (1 << k):
                        s += frac[others[k]]
                        # Early exit: if s already > target, skip (all positive)
                        if s > target:
                            break
                if s == target:
                    return False
        else:
            # For larger sets, use a randomized/heuristic check or meet-in-the-middle
            # Meet in the middle: split others into two halves
            half = m // 2
            left = others[:half]
            right = others[half:]
            # Compute all subset sums for left
            left_sums = set()
            lm = len(left)
            for mask in range(1 << lm):
                s = Fraction(0, 1)
                for k in range(lm):
                    if mask & (1 << k):
                        s += frac[left[k]]
                left_sums.add(s)
            # For each subset of right, check if target - right_sum is in left_sums
            rm = len(right)
            for mask in range(1 << rm):
                s = Fraction(0, 1)
                for k in range(rm):
                    if mask & (1 << k):
                        s += frac[right[k]]
                        if s > target:
                            break
                else:
                    if target - s in left_sums:
                        return False
            # Also check empty subset of right (mask=0 handled above as s=0)
            # Wait, mask=0 gives s=0, so we check if target in left_sums. That's covered.
            # But we need to make sure we don't use empty subset of BOTH (that would be 0 != target since target > 0)
            # Actually the problem says "subset of the OTHER selected elements" - can the subset be empty?
            # "sum of reciprocals of a subset" - empty subset sums to 0, and 1/x != 0, so it's fine.
    return True

def _is_valid_fast(S):
    """Faster validity check using integer arithmetic (multiply by LCM)."""
    S = list(S)
    n = len(S)
    if n == 0:
        return True
    # Use integer arithmetic: 1/a = sum(1/b_j) iff LCM * (1/a) = LCM * sum(1/b_j)
    # Let L = lcm of all elements. Then L/a is integer, L/b_j is integer.
    # So check: L/S[i] != sum of L/S[j] for j in subset
    from math import gcd
    from functools import reduce
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, S, 1)
    vals = [L // x for x in S]
    # Now check: for each i, vals[i] != sum of vals[j] for j in any non-empty subset of others
    # (empty subset sums to 0, vals[i] > 0, so no issue)
    for i in range(n):
        target = vals[i]
        others = [vals[j] for j in range(n) if j != i]
        m = len(others)
        if m <= 20:
            for mask in range(1, 1 << m):
                s = 0
                for k in range(m):
                    if mask & (1 << k):
                        s += others[k]
                        if s > target:
                            break
                if s == target:
                    return False
        else:
            # Meet in the middle
            half = m // 2
            left = others[:half]
            right = others[half:]
            left_sums = set()
            lm = len(left)
            for mask in range(1 << lm):
                s = 0
                for k in range(lm):
                    if mask & (1 << k):
                        s += left[k]
                left_sums.add(s)
            rm = len(right)
            for mask in range(1 << rm):
                s = 0
                for k in range(rm):
                    if mask & (1 << k):
                        s += right[k]
                        if s > target:
                            break
                else:
                    if target - s in left_sums:
                        return False
    return True

def construct():
    start = time.time()
    # Start with the top-half as a baseline
    best = list(range(13, 25))
    best_size = len(best)
    
    # Try greedy: start empty, add elements in order of largest reciprocal (smallest element) 
    # but that's bad. Better: start with large elements, try adding smaller ones.
    
    # Greedy: start with [24, 23, ..., 13] and try to add elements from 12 down to 1
    S = list(range(24, 12, -1))  # [24, 23, ..., 13]
    for candidate in range(12, 0, -1):
        new_S = S + [candidate]
        if _is_valid_fast(new_S):
            S = new_S
            if len(S) > best_size:
                best = S[:]
                best_size = len(S)
    
    # Local search: try swapping elements
    import random
    random.seed(42)
    
    current = S[:]
    current_valid = _is_valid_fast(current)
    
    while time.time() - start < DEADLINE:
        improved = False
        # Try adding a random element not in current
        if len(current) < 24:
            candidates = [x for x in range(1, 25) if x not in current]
            random.shuffle(candidates)
            for c in candidates:
                if time.time() - start > DEADLINE - 2:
                    break
                new_S = current + [c]
                if _is_valid_fast(new_S):
                    current = new_S
                    improved = True
                    if len(current) > best_size:
                        best = current[:]
                        best_size = len(current)
                    break
        
        # Try removing a random element and adding a different one
        if len(current) > 1:
            idx = random.randint(0, len(current) - 1)
            removed = current.pop(idx)
            candidates = [x for x in range(1, 25) if x not in current]
            random.shuffle(candidates)
            added = False
            for c in candidates:
                if time.time() - start > DEADLINE - 2:
                    break
                new_S = current + [c]
                if _is_valid_fast(new_S):
                    current = new_S
                    added = True
                    if len(current) > best_size:
                        best = current[:]
                        best_size = len(current)
                    break
            if not added:
                # Put it back
                current.append(removed)
                random.shuffle(current)
        
        # Try removing one element to see if we can add two
        if len(current) > 2 and not improved:
            idx = random.randint(0, len(current) - 1)
            removed = current.pop(idx)
            candidates = [x for x in range(1, 25) if x not in current and x != removed]
            random.shuffle(candidates)
            # Try adding one
            added_one = None
            for c in candidates:
                if time.time() - start > DEADLINE - 2:
                    break
                if _is_valid_fast(current + [c]):
                    added_one = c
                    break
            if added_one is not None:
                current.append(added_one)
                # Try adding another
                candidates2 = [x for x in range(1, 25) if x not in current]
                random.shuffle(candidates2)
                for c in candidates2:
                    if time.time() - start > DEADLINE - 2:
                        break
                    if _is_valid_fast(current + [c]):
                        current.append(c)
                        if len(current) > best_size:
                            best = current[:]
                            best_size = len(current)
                        break
    
    # Final: ensure best is valid
    if not _is_valid_fast(best):
        best = list(range(13, 25))
    
    return best
