"""Rewritten program using backtracking search with constraint propagation for difference triangle set."""
import time
import sys
from collections import deque

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all pairwise differences
    within rows are distinct across all rows.
    
    Uses backtracking search with heuristic ordering.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * n_rows] if k == 0 else []
    
    # Total number of differences needed: n_rows * C(k+1, 2)
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Counting bound: scope must be at least total_diffs
    # But we need to be careful, the differences are 1..scope, so scope >= total_diffs is a lower bound.
    # Actually, the max difference in a row is the last element.
    # The set of all differences must be a subset of {1, ..., max_entry}.
    # So max_entry >= total_diffs.
    
    best_solution = None
    best_scope = float('inf')
    
    # We will try to find a solution with scope S.
    # We can iterate S upwards from the lower bound.
    # Lower bound is total_diffs.
    # Upper bound: we can use a greedy heuristic to get an initial upper bound.
    
    def greedy_upper_bound():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return max(max(r) for r in rows)
    
    initial_ub = greedy_upper_bound()
    
    # We will search for solutions with scope S in increasing order.
    # If we find one, it's optimal for that S.
    # We stop if we find a solution or if we exceed time limit.
    
    start_time = time.time()
    time_limit = 100.0  # Leave some margin for 120s limit
    
    # Helper for backtracking
    # State: current row index, current row being built, used diffs set
    # We build rows one by one. Within a row, we build entries one by one.
    
    # To speed up, we can pre-calculate or use sets.
    # used_diffs is a global set across all completed rows and the current partial row.
    
    def solve_row(row_idx, current_row, used_diffs, current_scope):
        nonlocal best_solution, best_scope, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # All rows constructed
            scope = max(max(r) for r in best_solution) # Wait, best_solution is not updated here properly in this structure
            # Actually, we are building rows sequentially.
            # Let's restructure: we build a list of rows.
            # If we reach n_rows, we have a full solution.
            # The scope is the max of the last row's max? No, max of all rows.
            # Since we build rows with increasing scope constraints? No, we fix a global S.
            # If we fix S, we just need to ensure all entries <= S.
            # Then the scope is S (or less, but we are trying S).
            # Actually, if we find a solution with max entry M <= S, the scope is M.
            # But we are iterating S. The first S for which a solution exists is the minimal S?
            # Not necessarily, because a solution with max entry M might exist for S=M, but also for S>M.
            # We want the minimal M.
            # So iterating S from lower bound up is correct. The first valid S is the answer?
            # Yes, because if a solution exists with max entry M, it is a valid solution for any S >= M.
            # So the minimal S for which a solution exists is the minimal M.
            
            # We need to store the actual rows to return.
            # In this recursive structure, we need to pass the rows built so far.
            return True
            
        # Pruning: if current time > limit, stop
        if time.time() - start_time > time_limit:
            return False
            
        # Heuristic: Try to build the next row.
        # To minimize the global scope, we should try to keep entries small.
        # However, we are constrained by the global S in the outer loop.
        
        # Start with a new row [0]
        new_row = [0]
        # used_diffs_local is the set of diffs in the current row so far, plus global used_diffs?
        # No, the constraint is that ALL diffs across ALL rows are distinct.
        # So used_diffs is the global set of all diffs found so far.
        
        # We need to fill new_row with k more elements.
        # Let's use a helper to complete the current row.
        
        def complete_row(idx, cur_row, used_local_diffs):
            nonlocal used_diffs
            
            if time.time() - start_time > time_limit:
                return False
                
            if idx == k + 1:
                # Row is complete
                # Check if all diffs in this row are distinct from each other (done by construction?)
                # And distinct from global used_diffs.
                # In this approach, we add diffs to used_diffs as we go?
                # If we add to used_diffs, we must backtrack (remove) if we fail later.
                
                # Let's use a different strategy:
                # 1. Generate a candidate row.
                # 2. Check if its diffs are disjoint from global used_diffs and internally distinct.
                # 3. If yes, add to global used_diffs and recurse to next row.
                # 4. If no, backtrack.
                
                # This "generate and check" is simpler but might be slow if generation is not constrained.
                # Better: construct the row entry by entry, checking conflicts immediately.
                
                # Since we are in a recursive backtracking, let's do it entry by entry.
                
                # Actually, the current structure `solve_row` calls `complete_row`.
                # Let's merge them or make `complete_row` the main recursive step.
                
                # For now, let's assume `complete_row` successfully built a row.
                # We need to add its diffs to the global set.
                # But wait, if we modify `used_diffs` in place, we need to undo it.
                
                # Let's refactor:
                # The function `backtrack(row_idx, current_rows, used_diffs)`
                
                # This is getting complicated inside the nested function.
                # Let's restart the backtracking logic structure for clarity.
                
                return True # Placeholder
            
            # Choose next element
            # Previous element is cur_row[-1]
            # Next element must be > cur_row[-1] and <= current_scope
            # Also, the difference (new_val - any_prev) must not be in used_diffs (global + local)
            
            prev = cur_row[-1]
            for next_val in range(prev + 1, current_scope + 1):
                # Check differences with all previous elements in this row
                valid = True
                local_diffs = []
                for p in cur_row:
                    d = next_val - p
                    if d in used_local_diffs or d in used_diffs:
                        valid = False
                        break
                    local_diffs.append(d)
                
                if valid:
                    # Try this value
                    new_local_diffs = used_local_diffs | set(local_diffs)
                    if complete_row(idx + 1, cur_row + [next_val], new_local_diffs):
                        return True
            
            return False
        
        # Call complete_row to build the row
        # used_local_diffs starts empty
        # But we need to ensure the row's diffs don't conflict with global used_diffs.
        # This is handled in the loop above.
        
        # If a row is found, we need to add its diffs to the global used_diffs.
        # But `complete_row` doesn't return the row or its diffs.
        # We need to capture the successful row.
        
        # This nested approach is messy. Let's write a cleaner top-level backtracker.
        
        return False

    # Let's implement a clean backtracker.
    
    def backtrack(row_idx, current_rows, used_diffs, scope):
        nonlocal best_solution, best_scope, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # Solution found
            # Since we iterate scope from low to high, the first found is optimal
            best_solution = [r[:] for r in current_rows]
            best_scope = scope
            return True
        
        # Build row row_idx
        # Start with [0]
        row = [0]
        used_in_row = set()
        
        def build_row(pos, cur_row, local_diffs):
            nonlocal used_diffs
            
            if time.time() - start_time > time_limit:
                return False
            
            if pos == k + 1:
                # Row complete
                # Check if local_diffs are disjoint from used_diffs
                # We already checked during construction, but let's be sure.
                # Actually, we check against used_diffs in the loop below.
                
                # Add this row to current_rows
                current_rows.append(cur_row)
                # Update used_diffs
                old_used = used_diffs
                used_diffs = used_diffs | local_diffs
                
                if backtrack(row_idx + 1, current_rows, used_diffs, scope):
                    return True
                
                # Backtrack
                current_rows.pop()
                used_diffs = old_used
                return False
            
            # Try next value
            prev = cur_row[-1]
            for next_val in range(prev + 1, scope + 1):
                # Check diffs with all previous in cur_row
                valid = True
                new_local_diffs = set(local_diffs)
                for p in cur_row:
                    d = next_val - p
                    if d in new_local_diffs or d in used_diffs:
                        valid = False
                        break
                    new_local_diffs.add(d)
                
                if valid:
                    if build_row(pos + 1, cur_row + [next_val], new_local_diffs):
                        return True
            
            return False
        
        if build_row(1, row, set()):
            return True
        
        return False

    # Iterate scope from lower bound to initial upper bound
    # Lower bound is total_diffs
    # But we can tighten it.
    # For (7,5), total_diffs = 7 * 15 = 105.
    # Known best is 112.
    
    for scope in range(total_diffs, initial_ub + 1):
        # Try to find a solution with max entry <= scope
        # Note: The max entry in the solution might be less than scope, but we are fixing the bound.
        # If we find a solution, the actual max entry is <= scope.
        # Since we start from the lowest possible scope, the first solution found will have the minimal possible max entry?
        # Yes, because if a solution with max entry M exists, it is a valid solution for scope=M.
        # If we try scope < M, we will fail.
        # So the first scope that succeeds is the minimal M.
        
        # Reset state
        # used_diffs is empty
        # current_rows is empty
        if backtrack(0, [], set(), scope):
            return best_solution
            
        # If we timed out, we should return the best found so far?
        # But we haven't found any yet in this loop iteration if it returned False due to timeout.
        # We need to handle timeout globally.
        
        if time.time() - start_time > time_limit:
            break

    # If we exhausted the range or timed out without finding an optimal solution
    # Return the greedy solution as a fallback
    # Or the best found if we tracked it.
    # Since we didn't track partial bests in the backtracker (it only returns True/False),
    # and we iterate from low to high, if we fail, we don't have a solution for that scope.
    
    # Let's just return the greedy solution if backtracking fails.
    # To get a better fallback, we could run a quick local search or just return greedy.
    # Given the time limit, greedy is a safe bet.
    
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
