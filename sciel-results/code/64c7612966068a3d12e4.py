"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def _subset_sum_exists(vals, target):
    """Check if any non-empty subset of vals sums to target.
    Returns True if such a subset exists."""
    if not vals or target <= 0:
        return False
    
    # Filter and quick checks
    vals = [v for v in vals if 0 < v <= target]
    if not vals:
        return False
    if any(v == target for v in vals):
        return True
    if sum(vals) < target:
        return False
    
    n = len(vals)
    if n <= 20:
        # Meet-in-the-middle
        mid = n // 2
        left = vals[:mid]
        right = vals[mid:]
        
        left_sums = {0}
        for v in left:
            new_sums = set()
            for s in left_sums:
                ns = s + v
                if ns < target:
                    new_sums.add(ns)
                elif ns == target:
                    return True
            left_sums.update(new_sums)
        
        right_sums = {0}
        for v in right:
            new_sums = set()
            for s in right_sums:
                ns = s + v
                if ns < target:
                    new_sums.add(ns)
                elif ns == target:
                    return True
            right_sums.update(new_sums)
        
        for rs in right_sums:
            if rs <= target and (target - rs) in left_sums:
                if not (rs == 0 and target - rs == 0):
                    return True
        return False
    else:
        # For larger sets, use greedy heuristic + limited search
        # This is a conservative check - may miss some cases
        vals.sort(reverse=True)
        # Try to build up to target
        remaining = target
        for v in vals:
            if v <= remaining:
                remaining -= v
                if remaining == 0:
                    return True
        # Simple DP for smaller targets
        if target <= 10000:
            dp = [False] * (target + 1)
            dp[0] = True
            for v in vals:
                if v > target:
                    continue
                for s in range(target, v - 1, -1):
                    if dp[s - v]:
                        dp[s] = True
                        if s == target:
                            return True
            return dp[target]
        return False

def _is_safe_int(candidate, others):
    """Return True if 1/candidate is NOT equal to sum of 1/y for any non-empty
    subset of `others`."""
    if not others:
        return True
    
    L = candidate
    for y in others:
        L = _lcm(L, y)
    
    target_int = L // candidate
    if target_int <= 0:
        return True
    
    vals = [L // y for y in others]
    
    return not _subset_sum_exists(vals, target_int)

def _is_valid_full(subset):
    """Check if the entire subset is reciprocal-subsum-free."""
    s = set(subset)
    for x in s:
        others = list(s - {x})
        if not _is_safe_int(x, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin
    
    best_set = set()
    best_size = 0
    
    def update_best(s):
        nonlocal best_set, best_size
        if len(s) > best_size:
            best_set = set(s)
            best_size = len(s)
    
    # Strategy 1: Greedy from 24 down
    selected = set()
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if _is_valid_full(selected | {i}):
            selected = selected | {i}
    update_best(selected)
    
    # Strategy 2: Add back excluded
    for _pass in range(5):
        if time.time() > deadline:
            break
        excluded = sorted(set(range(1, 25)) - selected, reverse=True)
        added = False
        for x in excluded:
            if time.time() > deadline:
                break
            if _is_valid_full(selected | {x}):
                selected = selected | {x}
                added = True
                update_best(selected)
        if not added:
            break
    
    # Strategy 3: Multiple greedy orderings
    orderings = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [12, 24, 6, 18, 4, 20, 3, 21, 2, 22, 1, 23, 8, 16, 5, 15, 7, 14, 9, 13, 10, 11],
        [24, 20, 16, 12, 8, 4, 2, 18, 14, 10, 6, 22, 19, 15, 11, 7, 23, 9, 13, 5, 17, 3, 21, 1],
        [24, 12, 18, 6, 8, 16, 4, 20, 2, 10, 5, 15, 3, 19, 21, 7, 14, 9, 13, 11, 22, 23, 1, 17],
        [24, 21, 18, 15, 12, 9, 6, 3, 20, 17, 14, 11, 8, 5, 2, 22, 19, 16, 13, 10, 7, 4, 23, 1],
    ]
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        sel = set()
        for i in order:
            if time.time() > deadline - 2:
                break
            if _is_valid_full(sel | {i}):
                sel = sel | {i}
        update_best(sel)
    
    # Strategy 4: Local search with swaps
    if time.time() < deadline - 12:
        current = set(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            excluded = sorted(set(range(1, 25)) - current, reverse=True)
            current_list = sorted(current)
            
            for x in current_list:
                if time.time() > deadline - 5:
                    break
                for y in excluded:
                    if y in current:
                        continue
                    new_set = (current - {x}) | {y}
                    if len(new_set) <= best_size:
                        continue
                    if not _is_valid_full(new_set):
                        continue
                    temp = set(new_set)
                    for z in sorted(set(range(1, 25)) - temp, reverse=True):
                        if time.time() > deadline - 3:
                            break
                        if _is_valid_full(temp | {z}):
                            temp = temp | {z}
                    if _is_valid_full(temp) and len(temp) > best_size:
                        update_best(temp)
                        current = set(temp)
                        improved = True
                        break
                if improved:
                    break
    
    # Strategy 5: Base subsets
    if time.time() < deadline - 15:
        for start in range(12, 19):
            if time.time() > deadline - 5:
                break
            base = set(range(start, 25))
            if _is_valid_full(base):
                for i in range(start - 1, 0, -1):
                    if time.time() > deadline - 3:
                        break
                    if _is_valid_full(base | {i}):
                        base = base | {i}
                update_best(base)
    
    # Strategy 6: Random restarts
    if time.time() < deadline - 8:
        rng = random.Random(123)
        for trial in range(30):
            if time.time() > deadline - 3:
                break
            order = list(range(1, 25))
            rng.shuffle(order)
            sel = set()
            for i in order:
                if time.time() > deadline - 2:
                    break
                if _is_valid_full(sel | {i}):
                    sel = sel | {i}
            update_best(sel)
    
    if not best_set:
        best_set = {24}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
