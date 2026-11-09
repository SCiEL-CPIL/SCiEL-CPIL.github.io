"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def _is_valid(s):
        """Check if set s is reciprocal-subsum-free using LCM-based integer subset sum."""
        s_list = sorted(s)
        n = len(s_list)
        for i, x in enumerate(s_list):
            others = [y for y in s_list if y != x]
            if not others:
                continue
            # Compute LCM of x and all others
            L = x
            for y in others:
                L = L * y // gcd(L, y)
            target = L // x
            values = [L // y for y in others]
            
            # Quick check: if sum of all values < target, impossible
            if sum(values) < target:
                continue
            
            # Sort descending for better pruning
            values_sorted = sorted(values, reverse=True)
            m = len(values_sorted)
            
            # Suffix sums for pruning
            suffix = [0] * (m + 1)
            for j in range(m - 1, -1, -1):
                suffix[j] = suffix[j + 1] + values_sorted[j]
            
            # DFS with pruning
            found = [False]
            
            def dfs(idx, curr):
                if found[0] or time.time() > deadline:
                    return
                if curr == target:
                    found[0] = True
                    return
                if idx == m:
                    return
                if curr > target:
                    return
                if curr + suffix[idx] < target:
                    return
                # Try including values_sorted[idx]
                dfs(idx + 1, curr + values_sorted[idx])
                if found[0]:
                    return
                # Try excluding
                dfs(idx + 1, curr)
            
            dfs(0, 0)
            if found[0]:
                return False
        return True
    
    def _greedy_build(order):
        """Build a valid set by greedily adding elements in the given order."""
        s = set()
        for x in order:
            if time.time() > deadline:
                break
            candidate = s | {x}
            if _is_valid(candidate):
                s = candidate
        return s
    
    def _local_search(s):
        """Improve set s by trying to add elements not in s."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            for x in range(1, 25):
                if time.time() > deadline:
                    break
                if x in s:
                    continue
                candidate = s | {x}
                if _is_valid(candidate):
                    s = candidate
                    improved = True
        return s
    
    def _swap_improve(s):
        """Try removing one element and adding two others to increase size."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            for a in list(s):
                if time.time() > deadline:
                    break
                reduced = s - {a}
                for b in range(1, 25):
                    if time.time() > deadline:
                        break
                    if b in s:
                        continue
                    for c in range(b + 1, 25):
                        if time.time() > deadline:
                            break
                        if c in s:
                            continue
                        candidate = reduced | {b, c}
                        if len(candidate) > len(s):
                            if _is_valid(candidate):
                                s = candidate
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return s
    
    # Phase 1: Try multiple greedy orders
    best = set()
    
    # Order 1: Descending
    s1 = _greedy_build(range(24, 0, -1))
    if len(s1) > len(best):
        best = s1
    
    # Order 2: Ascending
    s2 = _greedy_build(range(1, 25))
    if len(s2) > len(best):
        best = s2
    
    # Order 3: Primes first
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    composites = [x for x in range(1, 25) if x not in primes and x > 1]
    s3 = _greedy_build(primes + composites)
    if len(s3) > len(best):
        best = s3
    
    # Order 4: Start with large numbers, then small
    order4 = list(range(24, 11, -1)) + list(range(1, 12))
    s4 = _greedy_build(order4)
    if len(s4) > len(best):
        best = s4
    
    # Phase 2: Local search on best
    best = _local_search(best)
    
    # Phase 3: Swap improvement
    best = _swap_improve(best)
    
    # Phase 4: Backtracking search
    # Use a smarter backtracking that processes elements in a fixed order
    # and uses the current best as a lower bound for pruning
    
    all_elements = list(range(24, 0, -1))
    best_size = len(best)
    best_set = set(best)
    
    def backtrack(index, current_set):
        nonlocal best, best_size, best_set
        if time.time() > deadline:
            return
        
        # Prune: if we can't beat best
        if len(current_set) + (len(all_elements) - index) <= best_size:
            return
        
        if index == len(all_elements):
            if len(current_set) > best_size:
                best = current_set
                best_size = len(best)
                best_set = set(best)
            return
        
        elem = all_elements[index]
        
        # Try including elem
        new_set = current_set | {elem}
        if _is_valid(new_set):
            backtrack(index + 1, new_set)
        
        # Try excluding elem
        backtrack(index + 1, current_set)
    
    # Only do backtracking if we have time
    if time.time() < deadline - 20:
        backtrack(0, set())
    
    # Ensure nonempty
    if not best:
        best = {24}
    
    return list(best)

# EVOLVE-BLOCK-END
