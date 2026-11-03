"""Difference triangle set solver using backtracking with conflict tracking."""
import time
import random

def construct(n_rows, k):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # We need n_rows rows, each with k+1 entries starting at 0, strictly increasing
    # All differences a_j - a_i (i < j) across all rows must be distinct
    # Minimize the maximum entry (scope)
    
    # Lower bound: we need n_rows * k(k+1)/2 distinct differences
    # Each difference is in [1, M] where M is the scope
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    best_result = None
    best_scope = float('inf')
    
    # First, try a quick greedy to get an initial upper bound
    greedy_result = _greedy_construct(n_rows, k)
    if greedy_result:
        best_result = greedy_result
        best_scope = max(max(row) for row in greedy_result)
    
    # Now try to improve with backtracking
    for scope in range(lower_bound, best_scope + 1):
        if time.time() > deadline:
            break
        
        result = _backtrack_construct(n_rows, k, scope, deadline)
        if result is not None:
            best_result = result
            best_scope = scope
            if scope == lower_bound:
                break
        elif time.time() > deadline:
            break
    
    if best_result is None:
        best_result = _greedy_construct(n_rows, k)
    
    return best_result


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes smallest available marks."""
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


def _backtrack_construct(n_rows, k, max_scope, deadline):
    """
    Backtracking search for n_rows valid rows with max entry <= max_scope.
    Returns list of rows or None if not found.
    """
    used_diffs = set()
    rows = []
    
    def dfs(row_idx):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        # Try to build row row_idx using backtracking
        row = [0]
        
        def build_row(pos, current_row):
            if time.time() > deadline:
                return None
            
            if pos == k + 1:
                return current_row[:]
            
            start_val = current_row[-1] + 1
            for val in range(start_val, max_scope + 1):
                new_diffs = {val - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                current_row.append(val)
                result = build_row(pos + 1, current_row)
                if result is not None:
                    return result
                current_row.pop()
            return None
        
        row = build_row(1, [0])
        if row is None:
            return False
        
        # Commit this row
        row_diffs = {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}
        used_diffs |= row_diffs
        rows.append(row)
        
        if dfs(row_idx + 1):
            return True
        
        # Backtrack
        rows.pop()
        used_diffs -= row_diffs
        return False
    
    if dfs(0):
        return rows
    return None
