"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from itertools import combinations
import time
import random

# EVOLVE-BLOCK-START

# LCM of 1..24
_LCM = 5354228880

def _recip_int(n):
    """Return 1/n as integer scaled by LCM."""
    return _LCM // n

def _is_valid_set(S):
    """Check if set S is reciprocal-subsum-free using integer arithmetic."""
    S_list = sorted(S)
    n = len(S_list)
    if n <= 1:
        return True
    
    # Optimization: The largest element a_max has the smallest reciprocal 1/a_max.
    # Any non-empty subset of the other reciprocals sums to a value strictly greater
    # than the smallest reciprocal in that subset. Since all other reciprocals are
    # larger than 1/a_max, their sum is > 1/a_max. Thus, 1/a_max can never be a subsum.
    # We only need to check elements a < max(S).
    
    for i in range(n - 1): # Skip the last (largest) element
        a = S_list[i]
        target = _LCM // a
        # others are S_list[0..i-1] and S_list[i+1..n-1]
        # To avoid creating a new list, we can iterate or slice
        # Slicing is O(n), which is fine.
        others = [_LCM // b for j, b in enumerate(S_list) if j != i]
        m = len(others)
        if m == 0:
            continue
        
        # Quick check: If sum of all others < target, impossible
        sum_others = sum(others)
        if sum_others < target:
            continue
            
        # Subset sum: can we form target from distinct elements of others?
        # Use meet-in-the-middle for larger sets
        if m <= 20:
            # Direct subset sum with set of reachable sums
            reachable = {0}
            for val in others:
                new_reachable = set()
                for s in reachable:
                    ns = s + val
                    new_reachable.add(ns)
                reachable |= new_reachable
                if target in reachable:
                    return False
        else:
            # Meet in the middle
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            left_sums = {0}
            for val in left:
                new_sums = set()
                for s in left_sums:
                    new_sums.add(s + val)
                left_sums |= new_sums
            
            right_sums = {0}
            for val in right:
                new_sums = set()
                for s in right_sums:
                    new_sums.add(s + val)
                right_sums |= new_sums
            
            # Check if target = l + r for some l in left_sums, r in right_sums
            # where not both are 0 (non-empty subset)
            right_set = set(right_sums)
            for l in left_sums:
                r = target - l
                if r in right_set:
                    # Check it's a non-empty subset
                    if l != 0 or r != 0:
                        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    best_set = set(range(13, 25))  # 12 elements
    
    # Greedy extension: start from high range, add lower elements
    base = set(range(13, 25))
    for order in [list(range(1, 13)), list(range(12, 0, -1))]:
        if time.time() > deadline:
            break
        current = set(base)
        for c in order:
            if time.time() > deadline:
                break
            if c in current:
                continue
            test = current | {c}
            if _is_valid_set(test):
                current = test
        if len(current) > len(best_set):
            best_set = current
    
    # Try different base sets
    for base_start in range(10, 18):
        if time.time() > deadline:
            break
        base = set(range(base_start, 25))
        if not _is_valid_set(base):
            continue
        current = set(base)
        candidates = [x for x in range(1, base_start)]
        for c in candidates:
            if time.time() > deadline:
                break
            test = current | {c}
            if _is_valid_set(test):
                current = test
        if len(current) > len(best_set):
            best_set = current
    
    # Greedy from scratch, high to low
    if time.time() < deadline:
        current = set()
        for x in range(24, 0, -1):
            if time.time() > deadline:
                break
            test = current | {x}
            if _is_valid_set(test):
                current = test
        if len(current) > len(best_set):
            best_set = current
    
    # Greedy from scratch, low to high
    if time.time() < deadline:
        current = set()
        for x in range(1, 25):
            if time.time() > deadline:
                break
            test = current | {x}
            if _is_valid_set(test):
                current = test
        if len(current) > len(best_set):
            best_set = current
    
    # Random orderings
    if time.time() < deadline - 5.0:
        rng = random.Random(12345)
        for trial in range(100):
            if time.time() > deadline:
                break
            order = list(range(1, 25))
            rng.shuffle(order)
            current = set()
            for x in order:
                test = current | {x}
                if _is_valid_set(test):
                    current = test
            if len(current) > len(best_set):
                best_set = current
    
    # Local search: try swapping elements
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for a in list(best_set):
                if time.time() > deadline:
                    break
                for b in range(1, 25):
                    if b in best_set:
                        continue
                    test = (best_set - {a}) | {b}
                    if len(test) == len(best_set):
                        if _is_valid_set(test):
                            best_set = test
                            improved = True
                            break
                if improved:
                    break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
