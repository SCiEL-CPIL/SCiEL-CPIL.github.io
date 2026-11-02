import sys
import time
import random

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Total differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on scope
    # The differences are 1..M. We need total_diffs distinct ones.
    # So M >= total_diffs.
    # Also, if the largest element is M, the largest diff is M.
    lower_bound = total_diffs
    
    # State
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    used_diffs = set()
    
    best_solution = None
    best_scope = float('inf')
    
    # We will fill rows one by one.
    # To break symmetry and find good solutions faster, we can try to keep rows "small".
    
    def get_scope():
        if not rows:
            return 0
        return max(max(r) for r in rows)
    
    def is_valid_addition(r, p, val):
        # r: row index, p: position index (1..k), val: candidate value
        # Check if val > rows[r][p-1]
        if val <= rows[r][p-1]:
            return False
            
        # Check differences
        for i in range(p):
            d = val - rows[r][i]
            if d in used_diffs:
                return False
        return True

    def backtrack(r, p):
        nonlocal best_solution, best_scope
        
        if time.time() > deadline:
            return False
        
        # If we are done with all rows
        if r == n_rows:
            scope = get_scope()
            if scope < best_scope:
                best_scope = scope
                best_solution = [row[:] for row in rows]
            return True
        
        # If we are done with current row, move to next
        if p == k + 1:
            return backtrack(r + 1, 1)
        
        # Determine the range of values to try
        # The value must be > rows[r][p-1]
        start_val = rows[r][p-1] + 1
        
        # Heuristic: Limit the search space.
        # The maximum possible value for this position shouldn't exceed the current best_scope.
        # Also, we can estimate a tighter upper bound.
        # The remaining differences to be generated in this row and future rows
        # must fit into the range [1, best_scope].
        
        # Pruning: 
        # 1. val cannot exceed best_scope (if we want to improve)
        # 2. The differences generated so far in this row + future rows must be distinct.
        
        # Let's try values from start_val to best_scope
        # To find the *smallest* scope, we should try smaller values first.
        
        # Optimization: If we have a current best_scope, we don't need to try values 
        # that make the current row's max exceed best_scope, UNLESS we haven't found any solution yet.
        # Actually, if we are looking for the optimal, we should try to find any solution first, then optimize.
        # But the greedy seed already finds a solution. We want to beat it.
        
        # Let's try values in increasing order.
        # Upper bound for val:
        # If we pick val, the max diff in this row will be at least val (since a0=0).
        # So val <= best_scope is a hard constraint if we want to improve.
        # If best_solution is None, we can go higher, but we should cap it to avoid infinite loops.
        # A reasonable cap is lower_bound * 2 or something.
        
        max_val_to_try = best_scope if best_solution is not None else lower_bound * 2
        
        for val in range(start_val, max_val_to_try + 1):
            if time.time() > deadline:
                return False
                
            if is_valid_addition(r, p, val):
                # Save state
                old_val = rows[r][p]
                rows[r][p] = val
                new_diffs = set()
                for i in range(p):
                    d = val - rows[r][i]
                    new_diffs.add(d)
                    used_diffs.add(d)
                    
                # Recurse
                if backtrack(r, p + 1):
                    # We found a better solution deeper? 
                    # No, backtrack returns True if it found a valid completion.
                    # We want to keep searching for better ones?
                    # Since we try values in increasing order, the first solution found for a given path 
                    # is the one with smallest local choices. But global scope depends on all rows.
                    # To find the global minimum, we should continue searching if we have time.
                    # However, if we found a solution, we can use its scope as the new upper bound.
                    pass # Don't return immediately, keep searching for better
        
                # Restore state
                rows[r][p] = old_val
                for d in new_diffs:
                    used_diffs.remove(d)
                    
        return False

    # Initial call
    backtrack(0, 1)
    
    if best_solution is None:
        # Fallback to greedy if backtracking failed or timed out without finding anything
        # (Unlikely for small n,k, but safe)
        used, rows_greedy = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows_greedy.append(row)
        return rows_greedy
        
    return best_solution
