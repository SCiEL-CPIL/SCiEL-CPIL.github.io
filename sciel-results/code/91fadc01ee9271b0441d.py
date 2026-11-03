"""Improved difference triangle set construction using backtracking with symmetry breaking."""
import time
import random

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # For the specific task: n_rows=4, k=7
    # We use a backtracking search with symmetry breaking
    
    best_rows = None
    best_scope = float('inf')
    
    # First, try greedy to get an initial upper bound
    greedy_rows = _greedy(n_rows, k)
    if greedy_rows is not None:
        best_rows = greedy_rows
        best_scope = max(max(row) for row in greedy_rows)
    
    # Now try to improve with backtracking
    # Symmetry breaking: row[i][1] > row[i-1][1] for i >= 1
    # This means the first element after 0 in each row is strictly increasing
    
    # We search rows one at a time, and within each row we backtrack
    # The key insight: we want to minimize the max entry, so we do a
    # breadth-first style search over the scope
    
    # For small n_rows and k, we can do a more structured search
    # Let's try iterative deepening on the scope
    
    # Counting bound: number of differences per row is k*(k+1)/2
    # Total differences = n_rows * k*(k+1)/2
    # These must all be distinct and <= scope
    # So scope >= n_rows * k*(k+1)/2
    lower_bound = n_rows * k * (k + 1) // 2
    
    # Try scopes from lower_bound upward
    for scope in range(lower_bound, best_scope + 1):
        if time.time() > deadline - 5:
            break
        
        rows = []
        used_diffs = set()
        prev_first = 0  # for symmetry breaking
        
        if _backtrack_rows(n_rows, k, scope, rows, used_diffs, prev_first, deadline):
            final_scope = max(max(row) for row in rows)
            if final_scope < best_scope:
                best_scope = final_scope
                best_rows = [list(r) for r in rows]
                break  # Found the minimum for this scope
    
    return best_rows if best_rows is not None else _greedy(n_rows, k)


def _greedy(n_rows, k):
    """Greedy construction for initial upper bound."""
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


def _backtrack_rows(n_rows, k, scope, rows, used_diffs, prev_first, deadline):
    """
    Backtrack to construct n_rows rows with max entry <= scope.
    rows: list of completed rows so far
    used_diffs: set of all differences used so far
    prev_first: the first element (after 0) of the previous row, for symmetry breaking
    """
    if len(rows) == n_rows:
        return True
    
    if time.time() > deadline - 2:
        return False
    
    # Try to construct the next row
    return _backtrack_row(k, scope, rows, used_diffs, prev_first, deadline)


def _backtrack_row(k, scope, rows, used_diffs, prev_first, deadline):
    """
    Backtrack to construct a single row of k+1 elements starting at 0,
    with max entry <= scope, using only unused differences.
    Symmetry: first element after 0 must be > prev_first.
    """
    row = [0]
    used_diffs_local = set(used_diffs)
    
    def backtrack(pos, current_max):
        nonlocal row, used_diffs_local
        
        if time.time() > deadline - 1:
            return False
        
        if pos == k + 1:
            rows.append(list(row))
            return True
        
        # Try next elements from current_max + 1 up to scope
        # For symmetry breaking on first element
        if pos == 1:
            start = prev_first + 1
        else:
            start = row[-1] + 1
        
        for x in range(start, scope + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs_local:
                continue
            
            # Pruning: check if we can still fit remaining elements
            remaining = k + 1 - (pos + 1)
            if remaining > 0:
                # We need 'remaining' more elements, each must be > x
                # The smallest possible next element is x+1
                # So we need at least x + remaining <= scope
                if x + remaining > scope:
                    continue
            
            row.append(x)
            used_diffs_local |= new_diffs
            
            if backtrack(pos + 1, x):
                return True
            
            row.pop()
            used_diffs_local -= new_diffs
        
        return False
    
    return backtrack(0, 0)

# EVOLVE-BLOCK-END
