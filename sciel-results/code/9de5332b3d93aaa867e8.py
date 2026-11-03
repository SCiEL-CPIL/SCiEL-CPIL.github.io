import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows and k+1 integers per row.
    Minimizes the largest entry.
    """
    # Parameters for the specific task: n_rows=4, k=7
    # But keep it general for the interface
    
    # Lower bound on m
    # Number of differences per row: C(k+1, 2)
    # Total differences: n_rows * C(k+1, 2)
    # These must be distinct integers in [1, m]
    min_m = n_rows * (k + 1) * k // 2
    
    # We search for the smallest m >= min_m that has a solution
    # Upper bound: The greedy solution or a known bound.
    # We'll iterate m upwards.
    
    # Time limit for the whole function
    start_time = time.time()
    max_time = 110  # Safety margin for 120s limit
    
    # We will try m from min_m upwards
    # To avoid getting stuck on an infeasible m that takes too long to prove infeasible,
    # we allocate a time budget per m check.
    # If min_m is 112, and we have 110s, we can afford to spend some time on 112.
    
    # Helper function to check if a solution exists for a given m
    def solve_for_m(m):
        used_diffs = 0  # Bitmask of used differences
        
        # rows[i] is the i-th row being constructed
        rows = []
        
        # To break symmetry, we enforce rows are lexicographically non-decreasing
        # rows[0] <= rows[1] <= rows[2] <= rows[3]
        
        def backtrack(row_idx, pos, current_row):
            nonlocal used_diffs
            
            # Time check
            if time.time() - start_time > max_time:
                return False, None
            
            if row_idx == n_rows:
                # All rows constructed
                return True, [r.copy() for r in rows]
            
            if pos == k + 1:
                # Current row is complete
                # Check symmetry: current_row >= previous row
                if row_idx > 0:
                    prev_row = rows[row_idx - 1]
                    # Lexicographic check
                    if current_row < prev_row:
                        return False, None
                
                rows.append(current_row.copy())
                success, result = backtrack(row_idx + 1, 0, [0])
                if success:
                    return True, result
                rows.pop()
                return False, None
            
            # Determine the range for current_row[pos]
            if pos == 0:
                min_val = 0
                max_val = 0
            else:
                min_val = current_row[pos - 1] + 1
                max_val = m
            
            # Iterate through possible values
            # Optimization: We can skip values that create immediate conflicts
            for v in range(min_val, max_val + 1):
                # Calculate new differences
                new_diffs = 0
                valid = True
                for i in range(pos):
                    d = v - current_row[i]
                    # d is in [1, m]
                    if (used_diffs >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                
                if not valid:
                    continue
                
                # Check if this value allows completion? (Basic bound)
                # Remaining elements: k + 1 - pos - 1
                # They must fit in [v+1, m]
                # Not a strong bound, but ensures v <= m - (remaining)
                remaining = k + 1 - pos - 1
                if v + remaining > m:
                    break # No more values possible for this pos
                
                # Recurse
                current_row.append(v)
                used_diffs |= new_diffs
                
                success, result = backtrack(row_idx, pos + 1, current_row)
                if success:
                    return True, result
                
                # Backtrack
                used_diffs ^= new_diffs
                current_row.pop()
            
            return False, None
        
        # Start backtracking
        success, result = backtrack(0, 0, [0])
        return success, result

    # Iterate m
    # We start from min_m.
    # If min_m is infeasible, we try min_m+1, etc.
    # To prevent hanging on infeasible small m, we can limit the time per m.
    # However, for small m, the search tree is pruned heavily, so it might be fast.
    # For larger m, it finds a solution fast.
    # The risk is an intermediate m that is infeasible but has a large search tree.
    
    # Let's try m from min_m to min_m + 50 (or until found)
    # If not found, fall back to a greedy construction to ensure validity.
    
    for m in range(min_m, min_m + 100):
        if time.time() - start_time > max_time - 5:
            break
            
        # Allocate a fraction of the remaining time to this m
        # Simple heuristic: if m is close to min_m, give it more time?
        # Or just use the global time limit.
        # If backtrack takes too long, it will return False due to time check.
        
        success, result = solve_for_m(m)
        if success:
            return result
    
    # If no solution found within the search range or time,
    # generate a valid (but potentially suboptimal) solution using greedy
    # to ensure we return something valid.
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
