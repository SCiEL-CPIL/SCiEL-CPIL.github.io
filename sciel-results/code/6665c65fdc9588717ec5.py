"""
Strategy: Constraint-Driven Backtracking with Binary Search on Scope + Local Search Polish.

Instead of simulated annealing over a noisy cost landscape, we:
1. Use a greedy construction to get an initial valid solution (upper bound on scope).
2. Binary search on the maximum entry (scope) to find the smallest feasible scope.
3. For each scope candidate, use constraint-driven backtracking to construct rows
   one by one, ensuring all differences are distinct globally.
4. After finding the minimal scope, apply local search to try to reduce scope further
   (though binary search should already find the minimum feasible).

Key structural differences from seed:
- Backtracking with global consistency checks (not local search on invalid states)
- Binary search on scope (systematic, not random exploration)
- Constraint propagation: when placing a value, we immediately check all new differences
  against the global used set
- Row construction order can be optimized (most constrained first)
"""
import random
import time
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple lower bound: need at least total_diffs distinct positive integers
    lower_bound = total_diffs
    
    # Helper to get max entry from rows
    def get_max_entry(rows):
        return max(max(row) for row in rows) if rows else 0
    
    # Helper to check if all differences are distinct
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    # Greedy construction: build rows one by one, placing smallest possible values
    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows
    
    # Backtracking row construction with given max_limit
    # Returns a valid row or None if not possible
    def backtrack_row(used_diffs, max_limit):
        """
        Construct a single row of k+1 increasing integers starting at 0,
        with all entries <= max_limit, such that all differences are not in used_diffs.
        Returns the row or None.
        """
        row = [0]
        
        def _bt(idx, last_val):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
            
            min_val = last_val + 1
            max_val = min(max_limit, min_val + (k - idx - 1))  # Prune: need room for remaining
            
            # Try values from min_val to max_val
            # Use randomized order for diversity
            candidates = list(range(min_val, max_val + 1))
            random.shuffle(candidates)
            
            for v in candidates:
                # Check if v creates any new differences that conflict
                new_diffs = set()
                conflict = False
                for a in row:
                    d = v - a
                    if d <= 0:
                        continue
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Check for conflicts within new_diffs itself (shouldn't happen since row is increasing)
                # Actually, differences within the same row to different elements are distinct
                # if the row is strictly increasing, so no issue there
                
                # Try this value
                row.append(v)
                used_diffs.update(new_diffs)
                
                if _bt(idx + 1, v):
                    return True
                else:
                    row.pop()
                    used_diffs.difference_update(new_diffs)
            
            return False
        
        if _bt(1, 0):
            return row[:]
        else:
            return None
    
    # Try to construct all rows with a given max_limit
    def try_construct(max_limit):
        """
        Try to construct n_rows valid rows with all entries <= max_limit.
        Returns the list of rows or None if impossible.
        """
        used_diffs = set()
        rows = []
        
        for r in range(n_rows):
            if time.time() > deadline:
                return None
            
            row = backtrack_row(used_diffs, max_limit)
            if row is None:
                return None
            rows.append(row)
        
        return rows
    
    # Local search: try to reduce max entry by moving individual elements down
    def local_search_polish(rows):
        """
        Given valid rows, try to reduce the maximum entry by moving individual
        elements to smaller values while maintaining validity.
        """
        if time.time() > deadline:
            return rows
        
        current_rows = [row[:] for row in rows]
        improved = True
        max_iter = 200  # Limit iterations
        
        while improved and max_iter > 0 and time.time() < deadline:
            improved = False
            max_iter -= 1
            
            # Try to move each element (except first which is 0) to a smaller value
            for r_idx in range(n_rows):
                if time.time() > deadline:
                    break
                for c_idx in range(k, 0, -1):  # Go from last to first
                    old_val = current_rows[r_idx][c_idx]
                    min_possible = current_rows[r_idx][c_idx - 1] + 1
                    
                    # Try values from min_possible to old_val - 1
                    for new_val in range(min_possible, old_val):
                        if time.time() > deadline:
                            break
                        
                        # Check if this move is valid
                        # Remove old differences involving old_val
                        old_diffs = set()
                        for i in range(len(current_rows[r_idx])):
                            if i != c_idx:
                                d = abs(current_rows[r_idx][i] - old_val)
                                if d > 0:
                                    old_diffs.add(d)
                        
                        # Check if new_val would create conflicts
                        new_diffs = set()
                        valid = True
                        for i in range(len(current_rows[r_idx])):
                            if i != c_idx:
                                d = abs(current_rows[r_idx][i] - new_val)
                                if d <= 0:
                                    valid = False
                                    break
                                new_diffs.add(d)
                        
                        if not valid:
                            continue
                        
                        # Check for conflicts with other rows
                        # We need to check all differences in the modified row
                        # against all differences in other rows
                        # This is expensive, so let's do a full check for now
                        test_rows = [row[:] for row in current_rows]
                        test_rows[r_idx][c_idx] = new_val
                        
                        if is_valid(test_rows):
                            current_rows[r_idx][c_idx] = new_val
                            old_val = new_val
                            improved = True
                            break
        
        return current_rows
    
    # Main algorithm
    # Step 1: Get initial solution with greedy
    initial_rows = pure_greedy()
    best_rows = [row[:] for row in initial_rows]
    best_max = get_max_entry(best_rows)
    
    # Step 2: Binary search on scope
    # We know scope is between lower_bound and best_max
    lo = max(lower_bound, 1)
    hi = best_max
    found_scope = best_max
    found_rows = best_rows
    
    # Try to find the minimum feasible scope using binary search
    # Each check uses backtracking which may be expensive
    # Limit the number of binary search steps
    max_bs_steps = 20
    step_count = 0
    
    while lo <= hi and step_count < max_bs_steps and time.time() < deadline:
        step_count += 1
        mid = (lo + hi) // 2
        
        # Try to construct with this scope
        # Use multiple random seeds for diversity in backtracking
        success = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            rows = try_construct(mid)
            if rows is not None:
                success = True
                found_scope = mid
                found_rows = rows
                break
        
        if success:
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Step 3: If we found a better scope, update best
    if found_rows is not None and get_max_entry(found_rows) < best_max:
        best_rows = found_rows
        best_max = get_max_entry(best_rows)
    
    # Step 4: Local search polish to try to reduce scope further
    if time.time() < deadline:
        polished = local_search_polish(best_rows)
        if polished is not None and is_valid(polished):
            polished_max = get_max_entry(polished)
            if polished_max < best_max:
                best_rows = polished
                best_max = polished_max
    
    # Final validation
    if not is_valid(best_rows):
        # Fallback to greedy if something went wrong
        best_rows = pure_greedy()
    
    return best_rows
# EVOLVE-BLOCK-END
