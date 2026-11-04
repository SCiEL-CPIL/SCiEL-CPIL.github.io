import time
import random
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = 145.0  # 5s safety margin from 150s limit
    
    # The problem: choose n_rows rows of (k+1) increasing integers starting at 0
    # such that all positive differences within each row are distinct across all rows.
    # Minimize the largest entry (scope).
    
    # For (7, 5): 7 rows, 6 elements each.
    # Each row has C(6,2) = 15 positive differences.
    # Total differences needed: 7 * 15 = 105.
    # Scope must be at least 105. Best known 112.

    def get_diffs(row):
        """Get all positive differences in a row as a set."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def backtrack_search(max_scope, time_limit):
        """
        Backtracking search for n_rows compatible rows with max entry <= max_scope.
        """
        start = time.time()
        used_diffs = set()
        rows = []
        
        def dfs(row_idx):
            if time.time() - start > time_limit:
                return "timeout"
            if row_idx == n_rows:
                return True
            
            def build_row_opt(current, start_val, local_diffs):
                if time.time() - start > time_limit:
                    return "timeout"
                if len(current) == k + 1:
                    # Validity already checked incrementally
                    rows.append(current[:])
                    used_diffs.update(local_diffs)
                    
                    result = dfs(row_idx + 1)
                    
                    if result is True:
                        return True
                    if result == "timeout":
                        return "timeout"
                    
                    rows.pop()
                    used_diffs.difference_update(local_diffs)
                    return False
                
                for x in range(start_val, max_scope + 1):
                    new_diffs = set()
                    valid = True
                    for a in current:
                        d = x - a
                        if d in used_diffs or d in local_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        current.append(x)
                        local_diffs.update(new_diffs)
                        res = build_row_opt(current, x + 1, local_diffs)
                        if res is True:
                            return True
                        if res == "timeout":
                            return "timeout"
                        current.pop()
                        local_diffs.difference_update(new_diffs)
                return False

            res = build_row_opt([0], 1, set())
            if res is True:
                return rows
            if res == "timeout":
                return "timeout"
            return None

    # Strategy:
    # 1. Try to find a solution with the theoretical lower bound (105 for 7,5).
    # 2. If that fails, try 106, 107, ... up to a reasonable limit (e.g., 120).
    # 3. If time is running out, return the best found so far (or None if no valid solution).
    
    # Calculate lower bound: n_rows * C(k+1, 2)
    lower_bound = n_rows * (k * (k + 1) // 2)
    
    best_rows = None
    best_scope = float('inf')
    
    # Let's try a range of scopes.
    # If lower bound is 105, try 105, 106, ...
    # But searching 105 might take long if no solution.
    # Let's set a time limit per scope attempt.
    
    current_time = time.time()
    remaining_time = deadline - current_time
    
    # Heuristic: If we have a known good bound for specific (n_rows, k), use it.
    # For (7,5), 112 is known.
    # For general, we don't know.
    
    # Let's try scopes from lower_bound to lower_bound + 30.
    # If we find one, great.
    
    max_scope_to_try = lower_bound + 30 # Reasonable upper bound for small n,k
    
    for scope in range(lower_bound, max_scope_to_try + 1):
        if time.time() > deadline - 5: # 5s buffer
            break
        
        time_for_attempt = min(10.0, deadline - (time.time() - start_time) - 2.0)
        if time_for_attempt < 1.0:
            break
            
        res = backtrack_search(scope, time_for_attempt)
        
        if res is not None and res != "timeout":
            best_rows = res
            best_scope = scope
            break # Found a valid solution for this scope, which is minimal for this search order.
            
        # If timeout or None, try next scope
        # Note: backtrack_search returns None if it exhausts search without finding solution.
        # It returns "timeout" if time limit hit.
        
    if best_rows is None:
        # If no solution found in the range, try a larger scope with less time?
        # Or just return a valid but larger solution?
        # The task requires minimizing the largest entry.
        # If we can't find a small one, we must find *a* valid one.
        # Let's try a greedy construction for a larger scope to ensure we return *something* valid.
        
        def greedy_large_scope():
            used_diffs = set()
            rows = []
            # Try to build rows with a larger scope, say 200
            scope = 200
            for _ in range(n_rows):
                found = False
                # Backtracking for one row
                def build_one_row(current, start_val, local_diffs):
                    if len(current) == k + 1:
                        rows.append(current[:])
                        used_diffs.update(local_diffs)
                        return True
                    
                    for x in range(start_val, scope + 1):
                        new_diffs = set()
                        valid = True
                        for a in current:
                            d = x - a
                            if d in used_diffs or d in local_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            current.append(x)
                            local_diffs.update(new_diffs)
                            if build_one_row(current, x + 1, local_diffs):
                                return True
                            current.pop()
                            local_diffs.difference_update(new_diffs)
                    return False
                
                if build_one_row([0], 1, set()):
                    continue
                else:
                    return None
            return rows

        best_rows = greedy_large_scope()
        if best_rows is None:
            # Last resort: return empty or raise? 
            # The contract says "invalid result ... scores 0". 
            # We must return a valid list.
            # If we can't find any, we are in trouble. 
            # But for (7,5) we should find one.
            return []

    return best_rows
