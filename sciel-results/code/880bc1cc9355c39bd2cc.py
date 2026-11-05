import time
import random

def construct(n_rows, k):
    # Function to run greedy with a random seed
    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
        return rows, max(max(r) for r in rows)

    # Run a few greedy attempts to find a good initial bound
    best_rows, best_max = run_greedy()
    for i in range(10):
        rows, m = run_greedy(i)
        if m < best_max:
            best_max = m
            best_rows = rows

    # If n_rows is 1, this is optimal
    if n_rows == 1:
        return best_rows

    deadline = time.time() + 110.0
    current_used = set()
    current_rows = []
    
    def dfs(row_idx, current_row):
        nonlocal best_max, best_rows, current_used, current_rows, deadline
        
        if time.time() > deadline:
            return
        
        if row_idx == n_rows:
            curr_max = max(max(r) for r in current_rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = [r[:] for r in current_rows]
            return

        if len(current_row) == k + 1:
            if row_idx > 0:
                if tuple(current_row) < tuple(current_rows[row_idx - 1]):
                    return
            
            current_rows.append(current_row[:])
            dfs(row_idx + 1, [])
            current_rows.pop()
            return

        last = current_row[-1]
        start = last + 1
        end = best_max
        
        for x in range(start, end + 1):
            valid = True
            new_diffs = set()
            for a in current_row:
                d = x - a
                if d in current_used:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            current_row.append(x)
            current_used.update(new_diffs)
            
            dfs(row_idx, current_row)
            
            current_row.pop()
            current_used.difference_update(new_diffs)
            
            if time.time() > deadline:
                return

    dfs(0, [0])
    return best_rows
