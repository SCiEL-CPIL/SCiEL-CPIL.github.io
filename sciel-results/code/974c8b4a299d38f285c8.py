def _build_row_backtrack(used_diffs, k, max_limit, node_limit=100000):
    """
    Backtracking search for a row of k+1 integers starting at 0, strictly increasing,
    with all pairwise differences distinct internally and not in used_diffs.
    Returns the row if found with max <= max_limit, else None.
    """
    row = [0]
    row_diffs = set()
    nodes_visited = [0]
    
    def backtrack(idx, last_val):
        nodes_visited[0] += 1
        if nodes_visited[0] > node_limit:
            return False
            
        if idx == k:
            return True
        
        for val in range(last_val + 1, max_limit + 1):
            conflict = False
            new_diffs = set()
            for i in range(len(row)):
                d = val - row[i]
                if d in row_diffs or d in used_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
            
            row.append(val)
            row_diffs.update(new_diffs)
            
            if backtrack(idx + 1, val):
                return True
            
            row.pop()
            row_diffs.difference_update(new_diffs)
            
        return False

    if backtrack(1, 0):
        return row
    return None
