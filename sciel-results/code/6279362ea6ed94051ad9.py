import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Total distinct differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for max entry M
    # The differences are distinct integers in [1, M].
    # We need total_diffs of them.
    # So M >= total_diffs.
    lower_bound = total_diffs
    
    # We will try M from lower_bound upwards
    # Limit the search to a reasonable range to avoid infinite loops if no solution exists
    max_search_range = 500 
    
    for M in range(lower_bound, lower_bound + max_search_range):
        if time.time() > deadline - 10.0: # Stop early if close to deadline
            break
            
        # State for backtracking
        used_mask = 0
        rows = []
        current_row = [0]
        
        def backtrack(row_idx, mark_idx):
            nonlocal used_mask, rows, current_row
            
            if time.time() > deadline:
                return False
                
            # Check if we have completed all rows
            if row_idx == n_rows:
                return True
                
            # If we are starting a new row
            if mark_idx == 0:
                # The first mark is always 0
                current_row = [0]
                return backtrack(row_idx, 1)
                
            # If we have completed the current row
            if mark_idx == k + 1:
                # Symmetry breaking check
                if row_idx > 0:
                    # Ensure current row's max is >= previous row's max
                    # And if equal, lexicographically >=
                    prev_max = max(rows[-1])
                    curr_max = max(current_row)
                    if curr_max < prev_max:
                        return False
                    if curr_max == prev_max and current_row < rows[-1]:
                        return False
                        
                rows.append(current_row[:])
                return backtrack(row_idx + 1, 0)
            
            # We need to choose the next mark for current_row
            # Current last mark
            last_mark = current_row[-1]
            
            # Candidate marks x must be > last_mark and <= M
            # Also, we must be able to complete the row.
            # Minimum remaining marks: k + 1 - mark_idx
            # The smallest possible increments are 1.
            # So min_possible_max_for_row = last_mark + (k + 1 - mark_idx)
            # If this > M, fail.
            if last_mark + (k + 1 - mark_idx) > M:
                return False
                
            # Iterate candidates
            # Heuristic: Try smaller x first to keep the row compact
            for x in range(last_mark + 1, M + 1):
                # Check differences
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    # d is at least 1
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if not valid:
                    continue
                
                # Recurse
                used_mask |= new_mask
                current_row.append(x)
                
                if backtrack(row_idx, mark_idx + 1):
                    return True
                    
                current_row.pop()
                used_mask ^= new_mask
                
            return False
        
        if backtrack(0, 0):
            # Found a valid solution with max M
            return rows

    # If no solution found within limits, return a greedy solution as fallback
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
