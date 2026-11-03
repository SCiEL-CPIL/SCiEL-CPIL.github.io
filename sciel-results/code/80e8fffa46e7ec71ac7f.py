"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import math
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct over all rows.
    Minimize the largest entry (scope).
    
    Uses backtracking with forward checking, constraint propagation,
    and symmetry breaking for the difference triangle set problem.
    """
    # Total number of differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound: smallest m such that m*(m+1)//2 >= total_diffs
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    lb = max(lb, k)
    
    start_time = time.time()
    timeout = 110  # Leave margin for 120s limit
    
    def solve(scope):
        """Try to construct n_rows rows with max entry <= scope."""
        used_diffs = bytearray(scope + 1)  # 1-indexed, used_diffs[d] = 1 if d is used
        used_count = 0
        rows = []
        
        def backtrack(row_idx, current_row):
            """
            row_idx: which row we're constructing (0-indexed)
            current_row: list of entries in current row (starts with [0])
            """
            nonlocal used_count
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete, move to next
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            # Determine the minimum next value
            min_next = current_row[-1] + 1 if current_row else 1
            max_next = scope
            
            # Symmetry breaking: for the first row, force the last element to be scope
            # This reduces the search space significantly
            if row_idx == 0 and len(current_row) == k:
                # Must pick exactly scope as the last element
                candidates = [scope]
            else:
                candidates = range(min_next, max_next + 1)
            
            # Heuristic: try values that create "popular" differences first
            # or use a specific ordering to find solutions faster
            # For now, try in order but with better pruning
            
            for x in candidates:
                # Check if adding x creates any duplicate differences
                new_diffs = []
                valid = True
                for a in current_row:
                    d = x - a
                    if used_diffs[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Forward checking: count available differences
                r_after = len(current_row) + 1
                rem = k + 1 - r_after
                # Diffs for each new entry in current row: r_after, r_after+1, ..., r_after+rem-1
                current_row_remaining_diffs = sum(range(r_after, r_after + rem))
                remaining_rows = n_rows - row_idx - 1
                future_rows_diffs = remaining_rows * diffs_per_row
                total_remaining_diffs = current_row_remaining_diffs + future_rows_diffs
                
                # Available differences in [1, scope] not in used_diffs
                available = scope * (scope + 1) // 2 - (used_count + len(new_diffs))
                
                if available < total_remaining_diffs:
                    continue
                
                # Additional pruning: check if the remaining entries can fit
                # The smallest possible values for remaining entries
                if rem > 0:
                    # We need 'rem' more entries, each > current_row[-1]
                    # The minimum total span needed
                    min_span = rem  # at least rem more units
                    if x + min_span > scope:
                        continue
                
                # Place x
                current_row.append(x)
                for d in new_diffs:
                    used_diffs[d] = 1
                used_count += len(new_diffs)
                
                if backtrack(row_idx, current_row):
                    return True
                
                # Undo
                current_row.pop()
                for d in new_diffs:
                    used_diffs[d] = 0
                used_count -= len(new_diffs)
            
            return False
        
        return backtrack(0, [0])
    
    # Try scopes starting from the lower bound
    for scope in range(lb, lb + 30):  # Try up to 30 scopes above the bound
        if time.time() - start_time > timeout:
            break
        
        result = solve(scope)
        if result:
            return rows
    
    # Fallback: use the greedy approach if backtracking fails or times out
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

# EVOLVE-BLOCK-END
