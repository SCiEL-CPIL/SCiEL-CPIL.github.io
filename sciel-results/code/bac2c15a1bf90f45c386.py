import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Greedy initialization
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        max_entry = max(max(r) for r in rows)
        return rows, max_entry

    best_rows, best_scope = greedy_construct()
    
    # If we already have a very good solution, maybe skip backtracking?
    # But we want to improve.
    
    # 2. Backtracking to improve
    # We want to find a solution with scope < best_scope.
    # So we prune if current_max >= best_scope.
    
    def backtrack(row_idx, current_rows, used_diffs, current_max):
        nonlocal best_scope, best_rows, deadline
        
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            if current_max < best_scope:
                best_scope = current_max
                best_rows = [r[:] for r in current_rows]
            return True
        
        # Symmetry breaking: row[i][1] > row[i-1][1]
        min_second = 1 if row_idx == 0 else current_rows[row_idx - 1][1]
        
        def construct_row(idx_in_row, row_so_far, max_so_far):
            nonlocal best_scope, best_rows, deadline
            
            if time.time() > deadline:
                return None
            
            if idx_in_row == k + 1:
                return row_so_far
            
            if idx_in_row == 1:
                start_val = min_second
            else:
                start_val = row_so_far[-1] + 1
                
            # Upper bound: v must be < best_scope
            # Also, we can stop if v is too large to fit in remaining slots?
            # Not strictly necessary, the loop bound handles it.
            
            for v in range(start_val, int(best_scope)):
                new_diffs = {v - u for u in row_so_far}
                
                if new_diffs & used_diffs:
                    continue
                
                new_max = max(current_max, v)
                if new_max >= best_scope:
                    break
                
                res = construct_row(idx_in_row + 1, row_so_far + [v], new_max)
                if res is not None:
                    return res
                    
            return None
        
        row_candidate = construct_row(1, [0], current_max)
        
        if row_candidate is not None:
            new_diffs = {row_candidate[j] - row_candidate[i] for i in range(len(row_candidate)) for j in range(i+1, len(row_candidate))}
            
            used_diffs.update(new_diffs)
            current_rows.append(row_candidate)
            
            if backtrack(row_idx + 1, current_rows, used_diffs, current_max):
                return True
            
            current_rows.pop()
            used_diffs.difference_update(new_diffs)
            
        return False
        
    backtrack(0, [], set(), 0)
    
    return best_rows

