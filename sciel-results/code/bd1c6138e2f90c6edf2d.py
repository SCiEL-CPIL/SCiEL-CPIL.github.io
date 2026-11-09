import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Lower bound on scope
    # Total diffs = n_rows * (k+1 choose 2)
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs  # Since diffs are 1..M, we need M >= total_diffs
    
    # Greedy initialization
    def greedy_construct():
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
                else:
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)
    
    # Backtracking search
    # rows: list of current row partials
    # used_diffs: set of all diffs used
    # row_idx: current row index
    # pos: current position in row
    # limit: max allowed value for this search
    
    def search(limit, rows, used_diffs, row_idx, pos):
        nonlocal best_rows, best_max
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            # Found a solution
            current_max = max(max(r) for r in rows)
            if current_max <= limit:
                best_rows = [list(r) for r in rows]
                best_max = current_max
            return True
            
        if pos == 0:
            # Start new row
            rows.append([0])
            if search(limit, rows, used_diffs, row_idx, 1):
                return True
            rows.pop()
            return False
            
        # Symmetry breaking: max of current row must be >= max of previous row
        # This is checked when the row is complete, but we can prune early
        # if the current last element is already > limit
        if rows[row_idx][-1] > limit:
            return False

        # If previous row exists, ensure final max >= prev max
        # We don't know final max yet, but we know it must be >= prev_max
        # This constraint is automatically satisfied if we allow any x, 
        # but it helps if we order x.
        
        last_val = rows[row_idx][pos - 1]
        
        # Try next values
        # Optimization: If pos == k (last element), we can bound x
        # We need x > last_val
        # Also x <= limit
        
        # Heuristic: Try x starting from last_val + 1
        # If we want to minimize max, we should try small x first?
        # Yes, DFS with small x first tends to find small max solutions.
        
        for x in range(last_val + 1, limit + 1):
            if time.time() > deadline:
                return False
            
            new_diffs = {x - a for a in rows[row_idx][:pos]}
            
            # Check if diffs are valid
            if new_diffs & used_diffs:
                continue
                
            # Check symmetry: if this is the last element of the row
            # and row_idx > 0, we must have x >= max(rows[row_idx-1])
            if pos == k and row_idx > 0:
                prev_max = max(rows[row_idx - 1])
                if x < prev_max:
                    # If x < prev_max, can we increase x later?
                    # No, pos == k is the last element.
                    # So this x is invalid for symmetry.
                    # But wait, if we skip this x, we try larger x.
                    # So we can just continue, or break if we assume x increases?
                    # x increases in the loop. So if x < prev_max, larger x might work.
                    # So continue.
                    continue

            # Valid x
            rows[row_idx].append(x)
            used_diffs |= new_diffs
            
            if search(limit, rows, used_diffs, row_idx, pos + 1):
                return True
                
            # Backtrack
            rows[row_idx].pop()
            used_diffs -= new_diffs
            
        return False

    # Search for better solutions
    # Iterate limit from lower_bound up to best_max - 1
    # If we find a solution with limit L, update best_max to L (or actual max)
    # And continue searching for L-1? No, we iterate limit downwards?
    # Or upwards?
    # If we find a solution with limit L, we know a solution exists with max <= L.
    # We want the minimum max.
    # So we should check limits from lower_bound upwards?
    # If limit 105 works, return.
    # If limit 105 fails, try 106.
    
    # However, if we find a solution at 112 (greedy), we know 112 works.
    # We want to check if 105, 106, ... work.
    
    for limit in range(lower_bound, best_max):
        if time.time() > deadline:
            break
        
        # Reset state for search
        rows = []
        used_diffs = set()
        
        # We need to pass state to search. 
        # The search function above modifies rows and used_diffs in place?
        # I used nonlocal and passed rows/used_diffs as args.
        # Let's make search pure or careful with state.
        
        # Actually, the previous search function signature was:
        # search(limit, rows, used_diffs, row_idx, pos)
        # It modifies rows and used_diffs.
        
        if search(limit, rows, used_diffs, 0, 0):
            # Found a solution with max <= limit
            # best_max is updated inside search
            # We can break if we found the theoretical lower bound?
            # Or continue to see if we can go lower?
            # The loop goes UP from lower_bound.
            # So if limit=105 works, we break? No, 105 is the min possible.
            # If limit=105 works, we have the optimum.
            if best_max <= limit:
                break
            # Otherwise, we found a solution at this limit, 
            # but maybe a smaller limit (which we already checked) failed?
            # Wait, the loop is `for limit in range(lower_bound, best_max)`.
            # It checks lower_bound first. If it works, best_max becomes lower_bound.
            # Then `if best_max <= limit` breaks.
            # If lower_bound fails, limit=lower_bound+1.
            # If that works, best_max becomes lower_bound+1.
            # We break because best_max <= limit.
            # This logic works.
            pass

    return best_rows
