import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def try_solve(limit, randomize=False):
        """Returns a valid DTS with max <= limit, or None."""
        
        def dfs(rows, current_row, used_diffs):
            if time.time() > deadline:
                return None
            
            # If current row is complete
            if len(current_row) == k + 1:
                # Check if we have all rows
                if len(rows) == n_rows - 1: 
                    # This is the last row
                    return rows + [current_row]
                
                # Start new row
                return dfs(rows + [current_row], [0], used_diffs)
            
            # If we are building a row
            last_val = current_row[-1]
            i = len(current_row)
            # Max value for current position to allow k-i more steps of size >= 1
            max_allowed = limit - (k - i)
            
            if max_allowed < last_val + 1:
                return None
            
            candidates = list(range(last_val + 1, max_allowed + 1))
            if randomize:
                random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = x - a
                    if d in used_diffs or d in new_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                result = dfs(rows, current_row + [x], used_diffs | new_diffs)
                if result:
                    return result
            
            return None
        
        return dfs([], [0], set())

    # Try deterministic search first for scope 134
    solution = try_solve(134, randomize=False)
    if solution:
        return solution
        
    # Try randomized restarts
    for i in range(100):
        if time.time() > deadline - 10:
            break
        random.seed(i)
        solution = try_solve(134, randomize=True)
        if solution:
            return solution
            
    # If still no solution, try higher limits
    for limit in range(135, 160):
        if time.time() > deadline - 5:
            break
        solution = try_solve(limit, randomize=False)
        if solution:
            return solution
        random.seed(limit)
        solution = try_solve(limit, randomize=True)
        if solution:
            return solution
            
    # Fallback: Greedy construction (may be invalid or high scope, but prevents crash)
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
