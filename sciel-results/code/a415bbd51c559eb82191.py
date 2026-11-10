import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def try_solve(limit):
        rows = []
        used_diffs = set()
        
        def dfs(rows, used_diffs):
            if time.time() > deadline:
                return None
            
            if len(rows) == n_rows:
                return [r[:] for r in rows]
                
            row = rows[-1]
            last_val = row[-1] if row else -1
            
            i = len(row)
            max_allowed = limit - (k - i)
            
            if max_allowed < last_val + 1:
                return None
                
            # Symmetry breaking: 
            # If this is not the first row, ensure the new row is lexicographically >= previous row?
            # Actually, it's easier to just sort the final rows.
            # But during search, we can enforce that the *first element* of the new row is >= first element of previous row?
            # No, all rows start with 0.
            # Enforce that the *second element* of the new row is >= second element of previous row?
            # This is a valid symmetry breaking for DTS if we assume rows are sorted by their second element.
            
            min_start = 1
            if len(rows) > 0:
                # Previous row is rows[-1]
                # Let's enforce that the current row's elements are >= previous row's elements at each position?
                # This is strong.
                # Or just: The sorted list of rows should be lexicographically sorted.
                # So, when adding a new row, it must be lexicographically >= the last added row.
                prev_row = rows[-1]
                # If current row is prefix of prev_row, it must be shorter? No, same length.
                # So, for each j, row[j] >= prev_row[j]?
                # This is a valid ordering.
                
                # Let's use this:
                # For the current row being built, at index j, the value must be >= prev_row[j] if j < len(row) is not applicable yet.
                # We are building row.
                # If i > 0, we have already chosen row[0..i-1].
                # We need to ensure that the final row will be >= prev_row.
                # This is hard to check mid-way.
                
                # Simpler symmetry breaking:
                # Fix the first row to be the "smallest" in some sense?
                # Or just don't bother. N=4 is small.
                pass
            
            for x in range(last_val + 1, max_allowed + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used_diffs:
                    continue
                
                # Additional pruning: 
                # If we have chosen some elements, can we still complete the row within the limit?
                # Yes, max_allowed handles the upper bound.
                
                rows.append(row + [x])
                used_diffs.update(new_diffs)
                
                result = dfs(rows, used_diffs)
                if result:
                    return result
                    
                rows.pop()
                used_diffs.difference_update(new_diffs)
                
            return None
            
        return dfs(rows, used_diffs)

    # Try to find a solution with scope 134
    solution = try_solve(134)
    if solution:
        # Sort rows to make them deterministic? Not required, but nice.
        solution.sort()
        return solution
        
    # If 134 failed, try higher
    for limit in range(135, 150):
        if time.time() > deadline - 5.0:
            break
        solution = try_solve(limit)
        if solution:
            solution.sort()
            return solution
            
    # Fallback: Greedy
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
    rows.sort()
    return rows
