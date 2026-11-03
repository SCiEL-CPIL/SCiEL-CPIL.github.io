import time
import sys
import random

def construct(n_rows, k):
    # Greedy initialization
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    best_rows = rows
    best_max = max(max(r) for r in rows)
    
    # Lower bound
    total_diffs = n_rows * (k * (k + 1) // 2)
    lower_bound = total_diffs
    
    # Time management
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Try to find a better solution using backtracking
    # We iterate M from lower_bound up to best_max - 1
    # If we find a solution with M < best_max, update best.
    
    # To speed up, we can skip M values if we know they are infeasible?
    # Hard to know.
    
    # We can also use the fact that if M is feasible, M+1 is feasible.
    # So we can binary search? No, we want the MINIMUM.
    # So linear search from LB is correct.
    
    # Optimization: If the greedy solution is already close to LB, 
    # the search space is small.
    
    for M in range(lower_bound, best_max):
        if time.time() > deadline:
            break
            
        # Try to find a solution with max element <= M
        # We use a backtracking solver
        # To avoid getting stuck in one M for too long, we set a sub-deadline
        sub_deadline = time.time() + 5.0 # 5 seconds per M
        
        # We can use a recursive function
        # To avoid recursion limit issues, we might need sys.setrecursionlimit
        # Depth is n_rows * (k+1) which is small.
        
        def backtrack(row_idx, current_row, local_diffs, global_diffs):
            nonlocal best_rows, best_max, deadline
            if time.time() > sub_deadline:
                return False
            
            if row_idx == n_rows:
                # Found a solution
                # Construct the full list of rows
                # We need to store the rows as we go
                # This implementation needs to track the full set of rows
                # Let's modify the approach: pass a list of completed rows
                pass
            
            # This simple structure is hard to manage state.
            # Let's use a class or a more careful closure.
            
        # Refined Backtracking Function
        # We will store the rows in a list `all_rows`
        # all_rows will contain the completed rows.
        # current_row is being built.
        
        all_rows = []
        
        def solve_row(idx, curr_row, l_diffs, g_diffs):
            if time.time() > sub_deadline:
                return False
            
            if idx == n_rows:
                # Success
                return True
            
            if len(curr_row) == k + 1:
                # Row complete
                # Check if it's the last row
                # Add to all_rows
                all_rows.append(curr_row)
                new_g_diffs = g_diffs.union(l_diffs)
                
                if idx == n_rows - 1:
                    return True
                else:
                    return solve_row(idx + 1, [0], set(), new_g_diffs)
            
            # Determine next element
            last = curr_row[-1]
            min_next = last + 1
            # Max next: 
            # We need to fit (k+1 - len(curr_row)) - 1 more elements after this one.
            # So if we pick x, we need x + (remaining_steps) <= M
            # remaining_elements = (k+1) - len(curr_row)
            # We are picking the first of these.
            # So x + (remaining_elements - 1) <= M
            remaining = (k + 1) - len(curr_row)
            max_next = M - (remaining - 1)
            
            if min_next > max_next:
                return False
                
            # Heuristic: try smaller values first
            # We can also randomize slightly if we want, but deterministic is fine for finding *a* solution
            for x in range(min_next, max_next + 1):
                # Check differences
                new_diffs = {x - a for a in curr_row}
                
                # Check conflicts with global and local
                if new_diffs & g_diffs:
                    continue
                if new_diffs & l_diffs:
                    continue
                
                # Valid
                new_curr_row = curr_row + [x]
                new_l_diffs = l_diffs.union(new_diffs)
                
                if solve_row(idx, new_curr_row, new_l_diffs, g_diffs):
                    return True
            
            return False
        
        # Start solving
        if solve_row(0, [0], set(), set()):
            # Solution found
            best_rows = all_rows
            best_max = M
            # Since we are iterating M upwards, the first solution found for a given M is valid.
            # But is it the best for that M? Yes, we just need *a* solution.
            # And since M is increasing, this is the global minimum M found so far.
            # We can break? 
            # No, we want the MINIMUM M. 
            # We are iterating M from LB upwards. 
            # So the first M for which a solution is found is the optimal M.
            # So we can break immediately.
            break
            
    return best_rows
