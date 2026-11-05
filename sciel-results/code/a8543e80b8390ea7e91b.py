def construct(n_rows, k):
    diff_count = n_rows * (k * (k + 1) // 2)
    lower_bound = diff_count
    
    result = []
    
    def solve_for_M(M):
        nonlocal result
        result = []
        node_count = 0
        max_nodes = 50000000 # Increased limit
        
        def dfs(row_idx, used_diffs, current_row, prev_max):
            nonlocal node_count
            node_count += 1
            if node_count > max_nodes:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Symmetry breaking: This row's max must be <= prev_max
                if current_row[-1] > prev_max:
                    return False
                
                result.append(current_row)
                # The max of this row becomes the new prev_max for the next row
                return dfs(row_idx + 1, used_diffs, [0], current_row[-1])
            
            last = current_row[-1]
            
            # Pruning: 
            # The current row must end with a value <= prev_max.
            # So x <= prev_max.
            # Also, we need to fit (k + 1 - len(current_row)) - 1 more marks after x.
            # The minimum possible value for the next mark is x + 1.
            # So the last mark will be at least x + (k + 1 - len(current_row) - 1).
            # This last mark must be <= prev_max.
            # So x + (k + 1 - len(current_row) - 1) <= prev_max
            # x <= prev_max - (k + 1 - len(current_row) - 1)
            
            remaining_marks = (k + 1) - len(current_row)
            min_last_mark_offset = remaining_marks - 1 # marks after x
            max_x = prev_max - min_last_mark_offset
            
            if max_x < last + 1:
                return False
                
            for x in range(last + 1, max_x + 1):
                # Quick check for smallest diff
                if (x - last) in used_diffs:
                    continue
                
                conflict = False
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                if dfs(row_idx, used_diffs | new_diffs, current_row + [x], prev_max):
                    return True
            return False
        
        # For the first row, prev_max is M
        return dfs(0, set(), [0], M)

    # Search for M
    for M in range(lower_bound, lower_bound + 100):
        if solve_for_M(M):
            return result
    
    # Fallback to Greedy
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
