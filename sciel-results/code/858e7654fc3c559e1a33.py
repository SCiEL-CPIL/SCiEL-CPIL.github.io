"""Difference triangle set (4,7) - Randomized Backtracking with Binary Search."""
import random

def _backtrack_mutable(row_idx, pos, rows, used_diffs, bound, node_limit, nodes_used):
    """
    Optimized backtracking with mutable set and node limit.
    Returns True if solution found, False otherwise.
    Modifies rows and used_diffs in-place.
    """
    n_rows = len(rows)
    k = len(rows[0]) - 1
    
    if nodes_used[0] > node_limit:
        return False

    if row_idx == n_rows:
        return True
        
    if pos == k:
        return _backtrack_mutable(row_idx + 1, 1, rows, used_diffs, bound, node_limit, nodes_used)
        
    current_row = rows[row_idx]
    last_val = current_row[pos - 1]
    
    start_val = last_val + 1
    end_val = bound
    
    # To add randomness, we can shuffle the order of values if we want.
    # But for finding *a* solution, order matters for speed.
    # Let's stick to increasing order for now, as it tends to find compact solutions.
    
    for val in range(start_val, end_val + 1):
        nodes_used[0] += 1
        if nodes_used[0] > node_limit:
            return False

        # Check if val - a_i is in used_diffs or in current row diffs
        valid = True
        new_diffs_to_add = []
        for i in range(pos):
            d = val - current_row[i]
            if d in used_diffs:
                valid = False
                break
            if d in new_diffs_to_add:
                valid = False
                break
            new_diffs_to_add.append(d)
            
        if valid:
            # Add to current row
            current_row[pos] = val
            # Add diffs to used_diffs
            for d in new_diffs_to_add:
                used_diffs.add(d)
                
            if _backtrack_mutable(row_idx, pos + 1, rows, used_diffs, bound, node_limit, nodes_used):
                return True
                
            # Backtrack: remove diffs
            for d in new_diffs_to_add:
                used_diffs.remove(d)
                
    return False

def _solve_for_bound(n_rows, k, bound, seed):
    """
    Try to find a solution where all values are <= bound.
    """
    random.seed(seed)
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    used_diffs = set()
    nodes_used = [0]
    node_limit = 100000 # Adjust if needed
    
    return _backtrack_mutable(0, 1, rows, used_diffs, bound, node_limit, nodes_used)

def _is_valid_solution(rows, k):
    """Check if the rows form a valid difference triangle set."""
    all_diffs = set()
    expected_count = len(rows) * k * (k + 1) // 2
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == expected_count

def _greedy_construct(n_rows, k):
    """Simple greedy fallback."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used_diffs):
                row.append(x)
                used_diffs |= new_diffs
            x += 1
        rows.append(row)
    return rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    lower_bound = n_rows * k * (k + 1) // 2
    upper_bound = 200
    
    best_rows = None
    best_bound = float('inf')
    
    current_bound = lower_bound
    found = False
    
    while current_bound <= upper_bound and not found:
        for seed in range(5):
            rows = [[0] * (k + 1) for _ in range(n_rows)]
            used_diffs = set()
            nodes_used = [0]
            
            success = _backtrack_mutable(0, 1, rows, used_diffs, current_bound, 50000, nodes_used)
            
            if success:
                if _is_valid_solution(rows, k):
                    best_rows = [row[:] for row in rows]
                    best_bound = current_bound
                    found = True
                    break
                else:
                    pass
        
        if not found:
            current_bound += 1
            
    if not found:
        best_rows = _greedy_construct(n_rows, k)
        
    return best_rows
