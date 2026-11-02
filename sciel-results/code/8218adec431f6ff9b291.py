"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct over all rows.
    Minimize the largest entry (scope).
    
    Uses backtracking with forward checking and constraint propagation.
    """
    # Total number of differences needed
    # Each row of size k+1 has C(k+1, 2) = k*(k+1)//2 differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # The minimum possible scope is at least the smallest m such that
    # the number of distinct positive differences in [1, m] is >= total_diffs
    # Number of distinct positive differences in [0, m] is m*(m+1)//2
    # But we also need each row to fit, so scope >= k
    # Simple lower bound: m such that m*(m+1)//2 >= total_diffs
    # Solve m^2 + m - 2*total_diffs >= 0
    # m >= (-1 + sqrt(1 + 8*total_diffs)) / 2
    import math
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    lb = max(lb, k)
    
    # We'll try scopes starting from lb upward
    # For each scope, we try to construct the rows using backtracking
    
    def solve(scope):
        """Try to construct n_rows rows with max entry <= scope."""
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, current_max):
            """
            row_idx: which row we're constructing
            current_row: list of entries in current row (starts with [0])
            current_max: max entry used so far in current_row
            """
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete, move to next
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0], 0):
                    return True
                rows.pop()
                return False
            
            # Determine the minimum next value
            min_next = current_max + 1 if current_row else 1
            max_next = scope
            
            # Try each possible next value
            for x in range(min_next, max_next + 1):
                # Check if adding x creates any duplicate differences
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Also check that new_diffs don't contain duplicates within themselves
                # (this can't happen since x is fixed and a are distinct, so x-a are distinct)
                
                # Forward checking: verify that it's still possible to complete
                # the current row and remaining rows with the updated used_diffs
                # Simple heuristic: count available differences
                updated_used = used_diffs | new_diffs
                
                # Check if we can still place the remaining entries in this row
                # and all subsequent rows
                remaining_in_row = k + 1 - len(current_row) - 1  # after adding x
                remaining_rows = n_rows - row_idx - 1  # after completing this row
                
                # Total differences still needed
                # Current row needs: remaining_in_row * (len(current_row) + 1) differences
                # Wait, let me think more carefully.
                # If current_row has r entries (after adding x, r = len(current_row)+1),
                # and we need k+1 total, we need k+1-r more entries.
                # Each new entry y creates differences y-a for all a in current_row.
                # So each new entry creates r differences (one for each existing entry).
                # Total diffs for current row completion: sum_{i=0}^{remaining_in_row-1} (r + i)
                # where r is the size after adding x
                
                r_after = len(current_row) + 1
                rem = k + 1 - r_after
                # Diffs for each new entry: r_after, r_after+1, ..., r_after+rem-1
                current_row_remaining_diffs = sum(range(r_after, r_after + rem))
                
                # Diffs for remaining rows: each has diffs_per_row
                future_rows_diffs = remaining_rows * diffs_per_row
                
                total_remaining_diffs = current_row_remaining_diffs + future_rows_diffs
                
                # Available differences in [1, scope] not in updated_used
                # Total possible diffs in [1, scope] is scope*(scope+1)//2
                available = scope * (scope + 1) // 2 - len(updated_used)
                
                if available < total_remaining_diffs:
                    continue
                
                # Place x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, current_row, x):
                    return True
                
                # Undo
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        return backtrack(0, [0], 0)
    
    # Try scopes starting from the lower bound
    # To avoid timeout, we limit the search
    start_time = time.time()
    timeout = 110  # Leave some margin
    
    for scope in range(lb, lb + 50):  # Try up to 50 scopes above the bound
        if time.time() - start_time > timeout:
            break
        
        # Reset state
        result = solve(scope)
        if result:
            return rows
    
    # Fallback: use the greedy approach if backtracking fails or times out
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
    return rows

# EVOLVE-BLOCK-END
