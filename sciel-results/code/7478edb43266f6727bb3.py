import sys
import time
import math
import numpy as np

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct over all rows.
    Minimize the largest entry (scope).
    
    Uses a CSP formulation with backtracking search, forward checking,
    and aggressive domain pruning via difference conflict detection.
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on scope
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    lb = max(lb, k)
    
    start_time = time.time()
    timeout = 110.0
    
    def solve(scope):
        """Try to construct n_rows rows with max entry <= scope."""
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row):
            """
            row_idx: which row we're constructing
            current_row: tuple of entries in current row (starts with (0,))
            """
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(list(current_row))
                if backtrack(row_idx + 1, (0,)):
                    return True
                rows.pop()
                return False
            
            # Determine valid next values
            min_next = current_row[-1] + 1 if current_row else 1
            max_next = scope
            
            # Pre-compute which values are invalid due to difference conflicts
            # For each candidate x, check if any (x - a) for a in current_row is in used_diffs
            # We can vectorize this check using numpy for speed
            
            # Try each possible next value, but with better ordering (smaller differences first
            # to pack tightly, or we could try values that create "rare" differences first)
            
            # For efficiency, let's pre-filter candidates
            candidates = []
            for x in range(min_next, max_next + 1):
                # Check conflicts with existing entries in current row
                conflict = False
                for a in current_row:
                    if (x - a) in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
            
            # Additional pruning: forward checking on remaining capacity
            # For each candidate, estimate if we can still complete
            if not candidates:
                return False
            
            # Try candidates in order
            for x in candidates:
                # Compute new differences
                new_diffs = frozenset(x - a for a in current_row)
                
                # Forward checking: can we still complete this row and future rows?
                r_after = len(current_row) + 1
                rem = k + 1 - r_after
                current_row_remaining_diffs = sum(range(r_after, r_after + rem))
                remaining_rows = n_rows - row_idx - 1
                future_rows_diffs = remaining_rows * diffs_per_row
                total_remaining_diffs = current_row_remaining_diffs + future_rows_diffs
                
                available = scope * (scope + 1) // 2 - len(used_diffs) - len(new_diffs)
                if available < total_remaining_diffs:
                    continue
                
                # Place x
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, current_row + (x,)):
                    return True
                
                # Undo
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        return backtrack(0, (0,))
    
    # Try scopes starting from the lower bound
    for scope in range(lb, lb + 80):
        if time.time() - start_time > timeout:
            break
        
        rows = []
        used_diffs = set()
        
        # We need a mutable container for rows and used_diffs in the closure
        # Let's restructure to use nonlocal or pass them differently
        
        result = solve(scope)
        if result:
            return rows
    
    # Fallback: greedy approach
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
