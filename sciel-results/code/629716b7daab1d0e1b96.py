def dfs(row_idx, current_row, current_row_diffs):
    if time.time() > deadline:
        return None
    
    if row_idx == n_rows:
        return rows
    
    if len(current_row) == 0:
        # Must start with 0
        current_row = [0]
        current_row_diffs = set()
        # Recurse immediately with this state
        return dfs(row_idx, current_row, current_row_diffs)
        
    if len(current_row) == k + 1:
        # Row complete
        rows.append(list(current_row))
        used_diffs.update(current_row_diffs)
        
        result = dfs(row_idx + 1, [], set())
        if result:
            return result
        
        rows.pop()
        for d in current_row_diffs:
            used_diffs.remove(d)
        return None
    
    last_val = current_row[-1]
    for x in range(last_val + 1, M_limit + 1):
        # ... check validity ...
        # ... recurse ...
