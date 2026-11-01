import time
import random

def construct(n_rows, k):
    # Total number of differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Greedy fallback
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x creates any new differences that are already used
                # Differences are x - a for a in row
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]: # Add new diffs
                        used.add(x - a)
            rows.append(row)
        return rows

    best_solution = greedy_construct()
    best_scope = max(max(row) for row in best_solution)
    
    # Backtracking search for optimal M
    # Start from counting bound
    min_scope = total_diffs
    
    # Set a deadline
    deadline = time.time() + 100.0
    
    # We will try M from min_scope upwards
    # If we find a solution for a specific M, it is the optimal M (since we start from bound)
    # We limit the range to avoid infinite loops if no solution exists (though one always exists)
    max_search_scope = min_scope + 50 
    
    def find_solution_with_scope(M):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, pos, current_row):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                rows.append(current_row)
                return backtrack(row_idx + 1, 0, [0])
            
            if pos == 0:
                # Must start with 0
                return backtrack(row_idx, pos + 1, [0])
            
            prev_val = current_row[-1]
            start_val = prev_val + 1
            
            # Optimization: If we can't fit the remaining marks in M, fail
            # We need (k+1 - pos) more marks.
            # The smallest possible values are start_val, start_val+1, ...
            # The largest will be at least start_val + (k - pos)
            # So we need start_val + (k - pos) <= M
            if start_val + (k - pos) > M:
                return False
            
            for x in range(start_val, M + 1):
                # Check differences
                conflict = False
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                # Add new differences
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, pos + 1, current_row + [x]):
                    return True
                
                # Undo
                used_diffs.difference_update(new_diffs)
            
            return False
        
        return backtrack(0, 0, [])

    # Try to find optimal M
    for M in range(min_scope, max_search_scope + 1):
        if time.time() > deadline:
            break
            
        # Reset rows list for this M attempt
        rows = []
        
        # We need to clear the rows list inside the function scope or pass it
        # The inner function uses `rows` from the enclosing scope
        # So we must clear it before each call
        rows.clear()
        
        if find_solution_with_scope(M):
            # Found a solution with scope M
            # This is the minimal M because we are iterating from the bound
            return rows
        
    # If we didn't find a better solution, return the greedy one
    return best_solution
