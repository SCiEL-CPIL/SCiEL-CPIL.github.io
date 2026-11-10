import sys
import time

def construct(n_rows, k):
    # k is the parameter such that row size is k+1
    # For the task (4,7), if it implies 8 integers, k=7.
    # If it implies 7 integers, k=6.
    # The prompt says "rows of 8 increasing integers" and "k+1 integers".
    # So if the evaluator passes k=7, we have 8 integers.
    
    def greedy(n_rows, k):
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

    initial_rows = greedy(n_rows, k)
    best_scope = max(max(row) for row in initial_rows)
    best_rows = [list(r) for r in initial_rows]
    
    # If n_rows or k is very small, greedy might be optimal or close.
    # For (4,7) [size 8], greedy is usually > 134.
    
    start_time = time.time()
    timeout = 115.0
    
    # Precompute total diffs per row
    diffs_per_row = k * (k + 1) // 2
    
    def dfs(row_idx, used_diffs, prev_max, current_rows):
        nonlocal best_scope, best_rows
        
        if time.time() - start_time > timeout:
            return False
        
        if row_idx == n_rows:
            scope = current_rows[-1][-1]
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in current_rows]
                return True
            return False
        
        # Pruning: Check if enough differences are available
        # Diffs needed for remaining rows (including current partial row completion)
        # Current row is not yet complete in this function call, 
        # but we pass the state of the *completed* previous rows.
        # The current row is built in the inner function.
        # So here, we have `row_idx` rows to build.
        # Diffs needed = n_rows_remaining * diffs_per_row
        # But wait, the current row is being built in `build_row`.
        # `dfs` is called *after* a row is complete? 
        # In the previous code, `dfs` called `build_row`.
        # Let's stick to that structure.
        
        def build_row(current_row, current_max, used_diffs_local):
            if time.time() - start_time > timeout:
                return False
            
            # Check if we can complete this row within best_scope
            # and if enough diffs remain for all future requirements.
            
            if len(current_row) == k + 1:
                # Row complete
                if current_max <= prev_max:
                    return False
                if current_max >= best_scope:
                    return False
                
                # Update used_diffs
                new_used = used_diffs_local.union({j - i for i in range(len(current_row)) for j in range(i+1, len(current_row))})
                
                # Check global diff availability for remaining rows
                rows_remaining = n_rows - (row_idx + 1)
                diffs_needed_future = rows_remaining * diffs_per_row
                available_diffs = best_scope - len(new_used)
                if diffs_needed_future > available_diffs:
                    return False
                
                return dfs(row_idx + 1, new_used, current_max, current_rows + [current_row])
            
            # Pruning: Estimate minimum possible max for this row
            # We need `rem = k + 1 - len(current_row)` more elements.
            # The smallest they can be is current_max + 1, current_max + 2, ...
            # So min_final_max = current_max + rem
            if current_max + (k + 1 - len(current_row)) >= best_scope:
                return False
            
            # Pruning: Difference availability for current row completion + future rows
            # Diffs already in current_row: len(current_row)*(len(current_row)-1)//2
            # Diffs needed for current row to complete: diffs_per_row - (len(current_row)*(len(current_row)-1)//2)
            # Diffs needed for future rows: (n_rows - row_idx - 1) * diffs_per_row
            # Total diffs needed from here:
            diffs_needed_current = diffs_per_row - (len(current_row) * (len(current_row) - 1) // 2)
            diffs_needed_future = (n_rows - row_idx - 1) * diffs_per_row
            total_needed = diffs_needed_current + diffs_needed_future
            available_diffs = best_scope - len(used_diffs_local)
            if total_needed > available_diffs:
                return False
            
            # Try next elements
            # Heuristic: Try x from current_max + 1
            # We can also limit x based on the fact that we need to fit remaining diffs.
            # But simple iteration is okay with pruning.
            
            for x in range(current_max + 1, best_scope):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs_local:
                    continue
                
                # Check if x allows completion
                # The new max is x.
                # We already checked min_final_max above, but x is the new current_max.
                
                if build_row(current_row + [x], x, used_diffs_local.union(new_diffs)):
                    return True
            return False
        
        return build_row([0], 0, used_diffs)

    # Start search
    dfs(0, set(), -1, [])
    
    return best_rows
