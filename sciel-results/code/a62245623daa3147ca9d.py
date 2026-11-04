import time

def construct(n_rows, k):
    """
    Solves the difference triangle set problem.
    n_rows: number of rows
    k: parameter such that each row has k+1 integers
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for the scope M
    lower_bound = total_diffs
    
    def solve_for_M(M):
        # M is the maximum allowed entry in any row
        # We use DFS with backtracking
        
        def build_row(row_idx, num_marks, current_row, used_mask, min_max_mark):
            # Check time
            if time.time() > deadline:
                return None
            
            # If row is complete
            if num_marks == k + 1:
                max_mark = current_row[-1]
                if max_mark < min_max_mark:
                    return None
                if row_idx == n_rows - 1:
                    return [current_row]
                else:
                    # Recurse to next row
                    # Next row starts with [0], num_marks=1
                    next_res = build_row(row_idx + 1, 1, [0], used_mask, max_mark)
                    if next_res is not None:
                        return [current_row] + next_res
                    return None
            
            # Determine range for next mark x
            last_mark = current_row[-1]
            start_x = last_mark + 1
            
            # If this is the last mark of the row, it must satisfy min_max_mark
            if num_marks == k:
                start_x = max(start_x, min_max_mark)
                
            # Upper bound for x:
            # We need to fit (k - num_marks) more marks after x.
            # The smallest they can be is x+1, x+2, ...
            # So the final max mark will be at least x + (k - num_marks).
            # This must be <= M.
            end_x = M - (k - num_marks)
            
            if start_x > end_x:
                return None
                
            # Pruning: Check if enough differences remain
            used_count = used_mask.bit_count()
            diffs_used_in_current_row = num_marks * (num_marks - 1) // 2
            diffs_remaining_in_current_row = diffs_per_row - diffs_used_in_current_row
            future_diffs_needed = (n_rows - row_idx - 1) * diffs_per_row
            total_needed = diffs_remaining_in_current_row + future_diffs_needed
            
            if used_count + total_needed > M:
                return None
            
            for x in range(start_x, end_x + 1):
                # Calculate new differences
                new_diffs = [x - a for a in current_row]
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                
                if new_mask & used_mask:
                    continue
                
                new_used_mask = used_mask | new_mask
                
                res = build_row(row_idx, num_marks + 1, current_row + [x], new_used_mask, min_max_mark)
                if res is not None:
                    return res
            
            return None
        
        # Start with row 0, 1 mark ([0]), no used diffs, min max mark 0
        return build_row(0, 1, [0], 0, 0)

    # Try increasing M from the lower bound
    # We limit the search range to avoid infinite loops if no solution exists in the bound
    # But a solution is guaranteed to exist for large enough M.
    # The greedy provides an upper bound.
    
    # Get a greedy solution first to have an upper bound and a fallback
    used, rows_greedy = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows_greedy.append(row)
    
    greedy_max = max(max(r) for r in rows_greedy)
    
    # Search for better M
    # Start from lower_bound, go up to greedy_max
    # If we find a solution, it's better than or equal to greedy.
    # Since we iterate M upwards, the first solution found is the minimal M.
    
    for M in range(lower_bound, greedy_max + 1):
        if time.time() > deadline - 5: # Stop if close to deadline
            break
        res = solve_for_M(M)
        if res is not None:
            return res
            
    # If no better solution found, return greedy
    return rows_greedy
