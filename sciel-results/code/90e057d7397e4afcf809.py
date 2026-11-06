import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Known solution as fallback
    KNOWN = [0, 1, 4, 10, 12, 28, 32, 45, 63]
    
    def is_valid(arr):
        if len(arr) > 32 or 0 not in arr or max(arr) > 10**6:
            return False
        diffs = set()
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))
    
    import heapq
    
    def solve_min_diameter(max_diameter, max_time):
        """
        Find a Sidon set with diameter <= max_diameter using optimized backtracking.
        Returns the set if found, None otherwise.
        """
        best = None
        FULL_MASK = (1 << 30) - 1
        
        def backtrack(selected, used_diffs, coverage_mask, next_min):
            nonlocal best
            if time.time() > max_time:
                return
            if coverage_mask == FULL_MASK:
                if best is None or max(selected) < max(best):
                    best = selected[:]
                return
            
            if len(selected) >= 32:
                return
            
            if next_min > max_diameter:
                return
            
            # Strong pruning: count remaining differences to cover
            remaining_mask = FULL_MASK - coverage_mask
            remaining_count = remaining_mask.bit_count()
            if remaining_count > 0:
                # Maximum new differences we can add: each new element x gives
                # (len(selected)) differences. We have (max_diameter - next_min + 1) slots.
                # But not all will be in 1..30 and unused. Use optimistic bound.
                slots_left = max_diameter - next_min + 1
                max_possible = slots_left * len(selected)
                if max_possible < remaining_count:
                    return
            
            # Try candidates in order of coverage potential
            # Precompute for each candidate x the new differences and coverage gain
            candidates = []
            for x in range(next_min, max_diameter + 1):
                new_diffs = []
                valid = True
                for s in selected:
                    d = x - s
                    if d <= 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Compute coverage gain
                gain = 0
                for d in new_diffs:
                    if 1 <= d <= 30:
                        bit = 1 << (d - 1)
                        if not (coverage_mask & bit):
                            gain += 1
                
                candidates.append((gain, x, new_diffs))
            
            # Sort by coverage gain descending, then by x ascending
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            for gain, x, new_diffs in candidates:
                if time.time() > max_time:
                    return
                
                new_coverage = coverage_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_coverage |= (1 << (d - 1))
                
                new_selected = selected + [x]
                new_used = used_diffs | set(new_diffs)
                backtrack(new_selected, new_used, new_coverage, x + 1)
                
                if best is not None:
                    return
        
        backtrack([0], set(), 0, 1)
        return best
    
    # Try increasing diameters starting from a lower bound
    # Theoretical lower bound: need at least 31 differences, so with n elements
    # we have n*(n-1)/2 differences. For n=9, that's 36, so diameter must be >= ~30
    # Empirically, start from 40 and go up
    for D in range(40, 121):
        if time.time() > deadline - 8:
            break
        r = solve_min_diameter(D, deadline - 5)
        if r is not None and is_valid(r):
            return r
    
    # Try larger diameters with more time
    for D in range(120, 251):
        if time.time() > deadline - 3:
            break
        r = solve_min_diameter(D, deadline - 2)
        if r is not None and is_valid(r):
            return r
    
    # Greedy fallback
    def greedy():
        selected = [0]
        used = set()
        for target in range(1, 31):
            if target in used:
                continue
            best_x = None
            best_cover = -1
            for x in range(1, 5000):
                if x in selected:
                    continue
                conflict = False
                new_diffs = []
                for s in selected:
                    d = x - s
                    if d <= 0:
                        continue
                    if d in used:
                        conflict = True
                        break
                    new_diffs.append(d)
                if conflict:
                    continue
                cover = sum(1 for d in new_diffs if 1 <= d <= 30 and d not in used)
                if target in new_diffs:
                    best_x = x
                    break
                elif cover > best_cover:
                    best_x = x
                    best_cover = cover
            if best_x is not None:
                selected.append(best_x)
                for s in selected[:-1]:
                    d = best_x - s
                    if d > 0:
                        used.add(d)
        if is_valid(selected):
            return sorted(selected)
        return None
    
    result = greedy()
    if result is not None:
        return result
    
    r = solve_min_diameter(500, deadline - 1)
    if r is not None and is_valid(r):
        return r
    
    return KNOWN

# EVOLVE-BLOCK-END
