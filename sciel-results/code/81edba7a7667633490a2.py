import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Greedy Baseline
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    best_rows = greedy()
    best_max_val = max(max(r) for r in best_rows)
    
    # 2. Backtracking Search to improve
    # We try to find solutions with smaller max values.
    # We iterate M downwards from best_max_val - 1.
    
    def backtrack_solve(M, time_limit):
        """
        Returns a list of rows if a valid configuration exists with max value <= M.
        Returns None if not found within time_limit.
        """
        used_diffs = set()
        rows = []
        
        # State for recursion:
        # row_idx: index of current row being built
        # current_row: list of elements in current row
        # last_val: last element in current_row (for x_start)
        
        # We use a helper to manage recursion depth and time
        import sys
        sys.setrecursionlimit(10000)
        
        def solve(row_idx, current_row, used_diffs):
            if time.time() > time_limit:
                return None
            
            if row_idx == n_rows:
                # All rows complete
                return rows + [current_row]
            
            if len(current_row) == k + 1:
                # Current row is full, move to next
                # We pass a copy of rows to avoid mutation issues in backtracking
                # But for efficiency, we can mutate and backtrack
                rows.append(current_row)
                res = solve(row_idx + 1, [0], used_diffs)
                if res is not None:
                    return res
                rows.pop()
                return None
            
            # Determine next x
            if not current_row:
                x_start = 1
            else:
                x_start = current_row[-1] + 1
            
            # Pruning:
            # We need 'rem' more elements in this row.
            rem = (k + 1) - len(current_row)
            
            for x in range(x_start, M + 1):
                if time.time() > time_limit:
                    return None
                
                # Check if x fits in the remaining space to end <= M
                # The smallest possible sequence for the remaining 'rem-1' elements after x
                # is x+1, x+2, ..., x+rem-1.
                # The last one is x + rem - 1.
                # Wait, 'rem' is the count of elements *to be added* including x?
                # No, 'rem' was calculated as total needed - current count.
                # So we are adding x now. After x, we need rem - 1 more.
                # The last element will be at least x + (rem - 1).
                if x + (rem - 1) > M:
                    break
                
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Recurse
                current_row.append(x)
                res = solve(row_idx, current_row, used_diffs | new_diffs)
                if res is not None:
                    return res
                current_row.pop()
                
            return None
        
        return solve(0, [0], used_diffs)

    # Linear search downwards
    # Start from best_max_val - 1
    # We give each step a time slice.
    # As M gets smaller, search becomes harder, so we can allocate more time?
    # Or just a fixed slice.
    # Let's say 2 seconds per M value.
    
    current_M = best_max_val - 1
    while current_M >= 112: # 112 is the counting bound
        if time.time() > deadline - 5.0: # Stop if close to deadline
            break
        
        # Allocate time for this M
        step_deadline = min(deadline - 1.0, time.time() + 2.0)
        
        sol = backtrack_solve(current_M, step_deadline)
        if sol is not None:
            # Found a better solution
            best_rows = sol
            best_max_val = max(max(r) for r in best_rows)
            # Continue to see if we can go even lower
            # Note: sol might have max < current_M, but that's fine.
            # We just continue to current_M - 1
        else:
            # No solution found (or timeout)
            # If timeout, we don't know. But if we assume no solution, we might miss it.
            # However, if it timed out, it's likely very hard.
            # We can break to save time for the final result.
            # Or continue?
            # If we continue, we might find a solution at M-1? Unlikely if M failed due to timeout.
            # If M failed due to "no solution", then M-1 also has no solution.
            # Since we can't distinguish, and timeout suggests hardness, let's break.
            # Actually, if it's a "no solution" case, it should be fast.
            # If it's a "timeout" case, it's slow.
            # Let's just break on timeout to be safe.
            # How to know if it timed out?
            # We can check if time.time() > step_deadline.
            if time.time() > step_deadline - 0.1:
                # Likely timed out
                break
            # Else, likely no solution, so M-1 will also have no solution.
            # So we can break.
            break

    return best_rows
