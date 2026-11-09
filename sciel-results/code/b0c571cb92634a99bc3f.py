import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    total_required_diffs = n_rows * (k + 1) * k // 2
    
    # Get an initial solution using greedy to set an upper bound
    used_g, rows_g = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used_g:
                row.append(x)
                used_g |= new
        rows_g.append(row)
    
    greedy_max = max(max(r) for r in rows_g)
    
    used_diffs = set()
    rows = []
    best_solution = None
    best_max = greedy_max
    
    def backtrack(row_idx, current_row):
        nonlocal used_diffs, best_solution, best_max
        
        # Pruning 1: Current max exceeds best
        if current_row:
            curr_max = current_row[-1]
            if curr_max >= best_max:
                return
        
        # Pruning 2: Counting bound
        remaining_required = total_required_diffs - len(used_diffs)
        available_slots = (best_max - 1) - len(used_diffs)
        
        if remaining_required > available_slots:
            return

        if len(current_row) == k + 1:
            # Row complete
            rows.append(current_row)
            
            if row_idx + 1 == n_rows:
                # All rows complete
                max_val = rows[-1][-1]
                if max_val < best_max:
                    best_max = max_val
                    best_solution = [r[:] for r in rows]
            else:
                # Start next row
                backtrack(row_idx + 1, [0])
            
            rows.pop()
            return
        
        # Try to extend current_row
        last_val = current_row[-1]
        
        for x in range(last_val + 1, best_max):
            new_diffs = {x - a for a in current_row}
            
            if new_diffs & used_diffs:
                continue
            
            used_diffs.update(new_diffs)
            current_row.append(x)
            
            backtrack(row_idx, current_row)
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)

    backtrack(0, [0])
    
    if best_solution:
        return best_solution
    else:
        return rows_g
