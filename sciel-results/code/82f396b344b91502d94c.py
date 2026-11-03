import time
import sys

def construct(n_rows, k):
    # Initial greedy solution
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x creates any new difference that is already used
                # Differences are x - a for a in row
                # We need to check if any (x - a) is in used
                # used is a bitset
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    # Mark new differences
                    for a in row:
                        d = x - a
                        used |= (1 << d)
                    row.append(x)
            rows.append(row)
        return rows, max(max(r) for r in rows)

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    initial_rows, initial_M = greedy()
    best_rows = initial_rows
    best_M = initial_M
    
    # Try to improve using backtracking
    # We try M from lower_bound to best_M - 1
    # Lower bound: total differences = n_rows * k * (k+1) // 2
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    # We can also use a tighter lower bound based on the specific structure, 
    # but total_diffs is a good start.
    
    # To avoid long runs on impossible M, we can start from a reasonable guess
    # Greedy is usually close. Let's try a few steps down from greedy.
    # If greedy is M_g, try M_g - 1, M_g - 2, ...
    
    # We'll iterate M downwards from best_M - 1 to lower_bound
    # But we should check time frequently.
    
    def backtrack(r, c, row, used):
        nonlocal best_rows, best_M
        nonlocal deadline, start_time
        
        if time.time() > deadline:
            return False, "timeout"
            
        if r == n_rows:
            # All rows constructed
            # Calculate max
            max_val = max(max(row) for row in best_rows) # Wait, best_rows is not updated here
            # Actually, we construct rows in a list
            return True, "success"
            
        if c == k + 1:
            # Row complete
            if r < n_rows - 1:
                # Start next row
                # Optimization: Sort rows? No, just proceed.
                # But we can enforce that the next row's last element is >= current row's last element?
                # Not necessarily, but it reduces symmetry.
                # Let's just proceed.
                return backtrack(r + 1, 1, [0], used)
            else:
                # All rows done
                return True, "success"
                
        # Try next value
        last_val = row[-1]
        min_next = last_val + 1
        max_next = best_M - 1 # We are trying to fit in best_M - 1? No, we pass M as limit.
        
        # Let's pass M_limit to the function
        # Actually, let's restructure to pass M_limit
        
        return False, "fail"

    # Restructured backtracking with M_limit
    def search(M_limit):
        nonlocal best_rows, best_M
        nonlocal deadline, start_time
        
        # State: r, c, row, used
        # We want to fill n_rows rows of length k+1
        
        def dfs(r, c, row, used):
            if time.time() > deadline:
                return None
                
            if r == n_rows:
                # Check if we actually filled all rows
                # This is called when r becomes n_rows after finishing the last row
                # Wait, my logic above was slightly off.
                # If c == k+1 and r == n_rows-1, we finish the last row.
                # Then we return success.
                pass
                
            if c == k + 1:
                if r < n_rows - 1:
                    # Move to next row
                    # To reduce symmetry, we can require that the last element of the new row 
                    # is at least the last element of the previous row?
                    # Or just start with [0].
                    # Let's try starting with [0] and see.
                    return dfs(r + 1, 1, [0], used)
                else:
                    # Finished all rows
                    # Construct the full list of rows
                    # We need to store the rows as we go.
                    # Let's pass a list of completed rows.
                    pass
            
            # Pruning: 
            # The remaining elements in this row must be at least last_val + 1, last_val + 2, ...
            # The last element will be at least last_val + (k + 1 - c)
            # If last_val + (k + 1 - c) > M_limit:
            #    return None
            
            last_val = row[-1]
            min_last = last_val + (k + 1 - c)
            if min_last > M_limit:
                return None
                
            # Try values
            for v in range(last_val + 1, M_limit + 1):
                # Check differences
                conflict = False
                new_diffs = []
                for a in row:
                    d = v - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs.append(d)
                    
                if conflict:
                    continue
                    
                new_used = used
                for d in new_diffs:
                    new_used |= (1 << d)
                    
                result = dfs(r, c + 1, row + [v], new_used)
                if result is not None:
                    return result
                    
            return None
        
        # We need to store the rows to return them.
        # Let's modify dfs to build the rows.
        
        rows = []
        
        def dfs2(r, c, row, used):
            if time.time() > deadline:
                return None
                
            if c == k + 1:
                rows.append(row)
                if r < n_rows - 1:
                    return dfs2(r + 1, 1, [0], used)
                else:
                    return rows[:] # Return a copy
                    
            last_val = row[-1]
            min_last = last_val + (k + 1 - c)
            if min_last > M_limit:
                return None
                
            for v in range(last_val + 1, M_limit + 1):
                conflict = False
                new_diffs = []
                for a in row:
                    d = v - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs.append(d)
                    
                if conflict:
                    continue
                    
                new_used = used
                for d in new_diffs:
                    new_used |= (1 << d)
                    
                result = dfs2(r, c + 1, row + [v], new_used)
                if result is not None:
                    return result
                    
            return None
        
        return dfs2(0, 1, [0], 0)

    # Iterate M downwards
    # Start from best_M - 1
    # If we find a solution, update best_M and best_rows
    # Continue to next lower M
    
    # To be efficient, we can try a few steps, then break if time is low.
    # We can also randomize the order of v? No, greedy order (smallest v first) is usually best for finding compact solutions.
    
    # Let's try M from best_M - 1 down to lower_bound
    # But we must be careful with time.
    
    current_M = best_M - 1
    while current_M >= lower_bound:
        if time.time() > deadline:
            break
            
        # Try to find a solution with max scope current_M
        # To avoid deep recursion issues, we rely on the time check.
        
        result = search(current_M)
        
        if result is not None:
            best_rows = result
            best_M = current_M
            # Try to go even lower
            current_M -= 1
        else:
            # No solution for this M, try next lower M? 
            # Wait, if no solution for M, there is no solution for M-1?
            # Yes, because the set of valid configurations for M-1 is a subset of those for M.
            # So if we fail for M, we can stop?
            # No, we are iterating downwards. 
            # If we fail for M, it means no solution exists with max scope M.
            # Then no solution exists with max scope M-1 either.
            # So we can break immediately!
            break
            
    return best_rows
