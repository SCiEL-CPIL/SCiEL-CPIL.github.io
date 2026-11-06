"""Rewritten program for large reciprocal-subsum-free sets."""
import time
import random
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_subsum_free(S, k):
    """
    Check if 1/k can be expressed as a sum of reciprocals of a non-empty
    subset of S. k is NOT in S when this is called.
    """
    if not S:
        return True
    
    target = Fraction(1, k)
    total = sum(Fraction(1, x) for x in S)
    
    if total < target:
        return True
    
    # Use meet-in-the-middle for larger sets
    n = len(S)
    if n <= 20:
        # Direct DFS with pruning
        elements = sorted(S, reverse=True)
        recips = [Fraction(1, x) for x in elements]
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + recips[i]
        
        found = False
        
        def dfs(i, current_sum, is_non_empty):
            nonlocal found
            if found:
                return
            if i == n:
                if current_sum == target and is_non_empty:
                    found = True
                return
            if current_sum + suffix_sums[i] < target:
                return
            # Exclude
            dfs(i + 1, current_sum, is_non_empty)
            if found:
                return
            # Include
            new_sum = current_sum + recips[i]
            if new_sum <= target:
                dfs(i + 1, new_sum, True)
        
        dfs(0, Fraction(0), False)
        return not found
    else:
        # Meet-in-the-middle for larger sets
        half = n // 2
        left = S[:half]
        right = S[half:]
        
        left_recips = [Fraction(1, x) for x in left]
        right_recips = [Fraction(1, x) for x in right]
        
        # Generate all subset sums for left half
        left_sums = set()
        for mask in range(1 << len(left)):
            s = Fraction(0)
            for j in range(len(left)):
                if mask & (1 << j):
                    s += left_recips[j]
            left_sums.add(s)
        
        # Check if any right subset sum + left subset sum == target
        for mask in range(1 << len(right)):
            s_right = Fraction(0)
            for j in range(len(right)):
                if mask & (1 << j):
                    s_right += right_recips[j]
            needed = target - s_right
            if needed in left_sums:
                # Need to ensure non-empty subset overall
                if mask != 0 or (needed != Fraction(0)):
                    return False
                elif mask == 0 and needed == Fraction(0):
                    # Empty subset, not valid
                    continue
        
        # Also check if target itself is in left_sums (right contributes nothing)
        if target in left_sums:
            # Need non-empty: check if target can be formed by non-empty left subset
            # target = 1/k, and left elements are >= 2 (since k is not in S)
            # If k >= 2, 1/k < 1/2 <= 1/x for x in left (x >= 2)
            # So we need at least 2 elements or specific combinations
            # Simple check: if target in left_sums and target != 0, it's a valid non-empty sum
            # unless the only way is empty set
            if target != Fraction(0):
                return False
        
        return True

def _is_valid_set(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    S_set = set(S)
    for k in S:
        others = [x for x in S if x != k]
        if not _is_subsum_free(others, k):
            return False
    return True

def _greedy_construct(order):
    """Greedy construction following the given order."""
    S = []
    for k in order:
        if _time_remaining() < 3:
            break
        others = S  # k not in S yet
        if _is_subsum_free(others, k):
            S.append(k)
    return S

def _local_search(S):
    """Try to improve S by swapping elements."""
    S_set = set(S)
    excluded = [x for x in range(1, 25) if x not in S_set]
    improved = False
    
    # Try adding excluded elements
    for e in excluded:
        if _time_remaining() < 3:
            break
        new_S = S + [e]
        if _is_valid_set(new_S):
            S = new_S
            S_set.add(e)
            improved = True
    
    # Try removing elements that cause conflicts and adding others
    if improved:
        return _local_search(S)
    
    return S

def construct():
    best_set = []
    
    # Strategy 1: Greedy descending (baseline)
    S1 = _greedy_construct(list(range(24, 0, -1)))
    if len(S1) > len(best_set):
        best_set = S1
    
    # Strategy 2: Greedy ascending
    if _time_remaining() > 10:
        S2 = _greedy_construct(list(range(1, 25)))
        if len(S2) > len(best_set):
            best_set = S2
    
    # Strategy 3: Multiple randomized greedy passes
    if _time_remaining() > 20:
        random.seed(42)
        for trial in range(50):
            if _time_remaining() < 10:
                break
            # Random ordering with bias
            order = list(range(1, 25))
            random.shuffle(order)
            S = _greedy_construct(order)
            if len(S) > len(best_set):
                best_set = S
    
    # Strategy 4: Local search from best greedy result
    if _time_remaining() > 15:
        S_improved = _local_search(list(best_set))
        if len(S_improved) > len(best_set):
            best_set = S_improved
    
    # Strategy 5: More randomized restarts if time allows
    if _time_remaining() > 30:
        random.seed(123)
        for trial in range(100):
            if _time_remaining() < 15:
                break
            order = list(range(1, 25))
            random.shuffle(order)
            S = _greedy_construct(order)
            if len(S) > len(best_set):
                best_set = S
                # Try local search on new best
                if _time_remaining() > 10:
                    S_imp = _local_search(list(best_set))
                    if len(S_imp) > len(best_set):
                        best_set = S_imp
    
    # Final: try a few more targeted approaches
    if _time_remaining() > 10 and len(best_set) < 20:
        # Try to build a set avoiding small numbers first
        for start in [5, 6, 7, 8]:
            if _time_remaining() < 5:
                break
            order = list(range(24, start - 1, -1)) + list(range(1, start))
            S = _greedy_construct(order)
            if len(S) > len(best_set):
                best_set = S
    
    if not best_set:
        return list(range(13, 25))
    
    return sorted(best_set)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
