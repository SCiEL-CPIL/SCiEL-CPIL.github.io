import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer
    
    # Parameters for backtracking
    # We will try to find a solution with scope <= best_scope
    # Initial best_scope can be high, we will refine it.
    # But we need a valid solution first.
    
    # Let's use a binary search on the scope?
    # Or just iterative deepening?
    # Given the small size (7x6), we can try to find the optimal directly with backtracking + pruning.
    
    # Heuristic: Cost of using a difference d is 1.0/d.
    # We want to minimize the total cost of used differences?
    # No, we just want to avoid conflicts.
    # The "degree" heuristic: choose the candidate that leaves the most options for the next step.
    # Approximation: Choose the candidate that uses the "least scarce" differences.
    # Scarcity is higher for small differences.
    
    best_result = None
    best_scope = float('inf')
    
    # We will run multiple restarts with different randomization seeds or slight heuristic variations
    # to escape local minima if backtracking gets stuck.
    
    # Pre-calculate potential candidates? No, dynamic.
    
    def backtrack(row_idx, current_row, used_diffs, current_max_val):
        nonlocal best_result, best_scope, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # All rows constructed
            scope = current_max_val
            if scope < best_scope:
                best_scope = scope
                best_result = [r[:] for r in rows] # Need to pass rows list
            return True
        
        # If we are building a row, current_row is the row being built
        # If current_row is empty, we start a new row.
        
        # Actually, the structure is:
        # We build row 0, then row 1, etc.
        # So state is: (row_idx, current_row, used_diffs)
        # If len(current_row) == k+1, we move to next row.
        
        if len(current_row) == k + 1:
            # Row complete
            if row_idx == n_rows - 1:
                # Last row
                scope = current_row[-1]
                if scope < best_scope:
                    best_scope = scope
                    # Copy rows
                    # We need to store the rows. 
                    # Let's pass a list of rows to backtrack.
                    return True
            else:
                # Start next row
                return backtrack(row_idx + 1, [0], used_diffs, current_max_val)
        
        # We need to choose the next element for current_row
        last_val = current_row[-1]
        
        # Determine upper bound for next element
        # It must be > last_val
        # It must be < best_scope (if we have a best_scope)
        upper_bound = best_scope - 1 if best_scope < float('inf') else 200
        
        candidates = []
        
        # Generate valid candidates
        # To optimize, we can limit the search space.
        # If best_scope is known, we don't go beyond it.
        
        for x in range(last_val + 1, upper_bound + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            # Heuristic Score:
            # We want to prefer candidates that use "large" differences (low 1/d sum)
            # This preserves small differences for future tight packing.
            cost = sum(1.0 / d for d in new_diffs)
            
            # Also, we want to keep x small to keep scope small.
            # There is a trade-off.
            # Let's combine: cost = sum(1.0/d) + lambda * x
            # Or just sort by cost, then by x.
            
            candidates.append((cost, x, new_diffs))
            
            # Limit candidates to avoid explosion
            if len(candidates) > 50:
                break
        
        if not candidates:
            return False
        
        # Sort candidates: Primary: cost (ascending - prefer large diffs), Secondary: x (ascending - prefer small scope)
        candidates.sort(key=lambda item: (item[0], item[1]))
        
        # Randomization: To avoid getting stuck in a specific local minimum,
        # we can shuffle the top candidates or pick randomly from top N.
        # But backtracking should handle it.
        # However, if the search space is huge, we might need to guide it.
        
        # Let's try the first few candidates.
        # To improve diversity, we can pick a random candidate from the top 10 with some probability.
        
        for cost, x, new_diffs in candidates:
            # Recurse
            if backtrack(row_idx, current_row + [x], used_diffs | new_diffs, max(current_max_val, x)):
                return True
        
        return False

    # The above backtrack function needs access to the `rows` list to update best_result.
    # Let's refactor to pass `rows` as a nonlocal or global variable, or return the solution.
    
    # Revised approach:
    # We will run a loop of restarts. In each restart, we run a backtracking search.
    # To make it fast, we will use a "failure" limit or time check.
    
    def solve_attempt(max_scope):
        nonlocal start_time, time_limit
        
        used_diffs = set()
        rows = []
        
        def _backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                # Check if we found a better solution
                # The scope is max of all rows' last elements
                # Since we build sequentially and enforce x < max_scope,
                # the current scope is max(r[-1] for r in rows)
                # But we don't know the final scope until all rows are done.
                # However, we are pruning with max_scope.
                # So if we complete, the scope is <= max_scope.
                
                # Calculate actual scope
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Upper bound for x
            # Must be < max_scope
            # Also, if we have a best_scope, we can prune if x >= best_scope
            upper_bound = min(max_scope - 1, best_scope - 1) if best_scope < float('inf') else max_scope - 1
            
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Heuristic: Cost = sum(1/d)
                # We prefer large diffs (small cost)
                cost = sum(1.0 / d for d in new_diffs)
                
                # Also consider the "tightness" of the next steps?
                # For now, just cost and x.
                
                candidates.append((cost, x, new_diffs))
                
                if len(candidates) > 30: # Limit branching
                    break
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda item: (item[0], item[1]))
            
            # Try candidates
            for cost, x, new_diffs in candidates:
                # Check if x is valid (it is, by construction)
                # Update state
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                if _backtrack(row_idx, current_row + [x]):
                    return True
                
                # Restore state
                used_diffs = old_used
                
        return _backtrack(0, [0])

    # Strategy:
    # 1. Try to find a solution with a reasonable initial scope (e.g., 150).
    # 2. If found, try to improve it by setting max_scope = best_scope - 1 and retrying.
    # 3. Repeat until time limit or no improvement.
    
    initial_scope = 200
    current_max_scope = initial_scope
    
    while time.time() - start_time < time_limit:
        # Reset state for this attempt
        # We need to clear best_result? No, we keep the global best.
        # But solve_attempt updates best_scope and best_result.
        
        # To avoid re-solving the same scope, we can track tried scopes?
        # Or just decrement max_scope if we found a solution.
        
        if solve_attempt(current_max_scope):
            # We found a solution with scope <= current_max_scope
            # best_scope is updated.
            # Let's try to improve by lowering the bound.
            if best_scope < current_max_scope:
                current_max_scope = best_scope - 1
            else:
                # If we found a solution at the boundary, maybe try a bit lower?
                # Or break if we hit the theoretical bound?
                # The counting bound is 105.
                if best_scope <= 105:
                    break
                current_max_scope = best_scope - 1
        else:
            # No solution found with current_max_scope
            # This implies the optimal scope is > current_max_scope?
            # No, it implies our search failed to find one within time or it doesn't exist.
            # If it doesn't exist, we should increase the scope?
            # But we are trying to minimize.
            # If we failed to find a solution with scope S, it might be because S is too small (optimal > S)
            # or because we got stuck.
            # If we got stuck, we should retry with randomization or different heuristics.
            # If S is too small, we should increase S.
            
            # How to distinguish?
            # If we explored the whole tree (unlikely for large S), it's too small.
            # If we timed out, it's too hard.
            
            # Let's just retry with a slightly larger scope if we think we might have missed it?
            # No, if we are minimizing, we start high and go down.
            # If we fail at S, it means no solution exists for <= S?
            # Not necessarily, just that our search didn't find one.
            
            # To be safe, if we fail, we can try increasing the scope slightly?
            # But we want to minimize.
            # If we started at 200 and found 112, we set current_max_scope = 111.
            # If we fail at 111, it means 111 is not possible (or search failed).
            # If 111 is not possible, the best is 112.
            # So we should stop or accept 112.
            
            # However, if the search failed due to time, we might miss a solution at 111.
            # We can retry a few times.
            
            # For now, let's assume if we fail, we break or retry with randomization.
            # Let's add a restart counter.
            
            # Actually, the `solve_attempt` function is deterministic.
            # We should add randomization to the candidate selection in `solve_attempt`.
            
            break # Simple strategy: if we can't improve, stop.

    if best_result is None:
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
        
    return best_result
