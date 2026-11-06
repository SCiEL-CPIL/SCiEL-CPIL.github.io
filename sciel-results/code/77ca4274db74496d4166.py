"""
Sidon set containing 0 with all differences 1..30, minimizing max element.
Uses a backtracking search with strong pruning.
"""

import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit
    
    # We need at most 32 elements, including 0.
    # We want to cover differences 1..30.
    
    # Let's try to find a set S = {0, a1, a2, ..., ak} where k <= 31
    # such that all positive differences are distinct (Sidon) and 1..30 are covered.
    
    # Key insight: A Sidon set with n elements has at most n(n-1)/2 distinct differences.
    # We need at least 30 differences, so n(n-1)/2 >= 30 => n >= 9 (since 8*7/2=28, 9*8/2=36).
    # So we need at least 9 elements (including 0).
    
    # Let's use a backtracking approach:
    # State: current set S (sorted), used_differences (set)
    # Goal: cover all differences 1..30
    
    # Optimization: Try to add elements in increasing order.
    # For a candidate element x > max(S), the new differences are x - s for s in S.
    # All these must be unused, and at least one should be in 1..30 if possible.
    
    best_solution = None
    best_max_val = float('inf')
    
    def solve():
        nonlocal best_solution, best_max_val
        
        # This is a recursive search
        # We'll build the set incrementally
        
        S = [0]
        used_diffs = set()
        
        def backtrack(S, used_diffs, remaining_needed):
            nonlocal best_solution, best_max_val
            
            if time.time() > deadline:
                return False
            
            # Check if we've covered all required differences
            if not remaining_needed:
                max_val = max(S)
                if max_val < best_max_val:
                    best_max_val = max_val
                    best_solution = list(S)
                return True
            
            # Pruning: if even adding all possible future elements can't cover remaining
            # differences, prune. But this is complex, so let's use a simpler bound.
            
            # Upper bound on how many more differences we can get from adding one element:
            # Adding x gives |S| new differences (x - s for each s in S)
            # We need len(remaining_needed) more differences
            # So we need at least ceil(len(remaining_needed) / |S|) more elements
            # Each new element adds at most its index (if we number them) to the max
            
            # Simple pruning: if max(S) + some_bound < best_max_val, we can still improve
            # But we don't know the bound easily.
            
            # Try adding new elements in increasing order
            current_max = max(S) if S else 0
            used = set(S)
            
            # What's the smallest difference we still need?
            min_needed = min(remaining_needed) if remaining_needed else None
            
            # Try candidate values for the next element
            # Start from current_max + 1
            # But also consider values that would create needed differences
            
            # Strategy: for each needed difference d, and each existing s in S,
            # x = s + d is a candidate. Also x = s - d if s - d > 0 and not used.
            # Collect candidates, sort, and try them.
            
            candidates = set()
            for d in remaining_needed:
                for s in S:
                    x = s + d
                    if x > current_max and x not in used:
                        candidates.add(x)
                    if d > 0:
                        x2 = s - d
                        if 0 < x2 and x2 not in used and x2 > current_max:
                            candidates.add(x2)
            
            # Also try current_max + 1, current_max + 2, etc. up to some limit
            # But limit by best_max_val
            upper = min(best_max_val - 1, current_max + 100)  # limit candidates
            for x in range(current_max + 1, upper + 1):
                if x not in used:
                    candidates.add(x)
            
            # Sort candidates
            candidates = sorted(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    return False
                
                # Check if adding x keeps Sidon property
                new_diffs = [x - s for s in S]
                # All new_diffs must be positive (x > all s) and not in used_diffs
                valid = True
                for nd in new_diffs:
                    if nd <= 0 or nd in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Also check that new_diffs are all distinct among themselves
                # (they are, since x - s_i are distinct for distinct s_i)
                
                # Update state
                new_used_diffs = used_diffs | set(new_diffs)
                new_remaining = remaining_needed - set(new_diffs)
                
                # Recurse
                result = backtrack(S + [x], new_used_diffs, new_remaining)
                if result:
                    return True
                
                # Backtrack
            
            return False
        
        backtrack(S, used_diffs, set(range(1, 31)))
        return best_solution
    
    # Initial attempt
    result = solve()
    
    if result is None or time.time() > deadline - 5:
        # Fallback: greedy approach
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        result = a
    
    return result
