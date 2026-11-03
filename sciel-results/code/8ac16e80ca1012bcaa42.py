"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _all_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all positive differences are distinct across all rows."""
    all_diffs = set()
    for row in rows:
        row_diffs = _all_diffs(row)
        if row_diffs & all_diffs:
            return False
        all_diffs |= row_diffs
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_row(used, k, max_val):
    """Build one row greedily given used differences and max_val bound."""
    row = [0]
    for _ in range(k):
        found = False
        for x in range(row[-1] + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _backtrack_row(used, k, max_val, row=None, start=1):
    """Build one row via backtracking given used differences and max_val bound."""
    if row is None:
        row = [0]
    if len(row) == k + 1:
        return row
    for x in range(start, max_val + 1):
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            new_used = used | new_diffs
            result = _backtrack_row(new_used, k, max_val, row + [x], x + 1)
            if result is not None:
                return result
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    # First try to find a solution with increasing max_val
    # Start from a reasonable lower bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    # m must be at least total_diffs_needed (each diff is in [1, m])
    lower_bound = total_diffs_needed
    
    best_rows = None
    best_scope = None
    
    # Try greedy first to get an initial solution
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, 200)
        if row is None:
            break
        rows.append(row)
        used |= _all_diffs(row)
    
    if len(rows) == n_rows:
        best_rows = rows
        best_scope = _scope(rows)
    else:
        # Need backtracking
        best_scope = 200
    
    # Try to improve using backtracking with increasing max_val
    if best_scope is not None:
        # Try smaller max values
        for max_val in range(lower_bound, best_scope):
            if time.time() > deadline:
                break
            used = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = _backtrack_row(used, k, max_val)
                if row is None:
                    success = False
                    break
                rows.append(row)
                used |= _all_diffs(row)
            if success:
                best_rows = rows
                best_scope = max_val
                break
    
    # Now try local search to improve
    if best_rows is not None and time.time() < deadline:
        best_scope = _scope(best_rows)
        
        # Local search: try to reduce individual entries
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for r_idx in range(n_rows):
                for e_idx in range(1, len(best_rows[r_idx])):
                    orig_val = best_rows[r_idx][e_idx]
                    # Try reducing this entry
                    for new_val in range(best_rows[r_idx][e_idx-1] + 1, orig_val):
                        if time.time() > deadline:
                            break
                        new_row = best_rows[r_idx][:]
                        new_row[e_idx] = new_val
                        # Check if row is still strictly increasing
                        if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                            continue
                        # Check if all differences are still valid
                        new_rows = best_rows[:]
                        new_rows[r_idx] = new_row
                        if _is_valid(new_rows):
                            new_scope = _scope(new_rows)
                            if new_scope < best_scope:
                                best_rows = new_rows
                                best_scope = new_scope
                                improved = True
                            break
    
    if best_rows is None:
        # Fallback: greedy with large bound
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, 500)
            if row is not None:
                rows.append(row)
                used |= _all_diffs(row)
        best_rows = rows
    
    return best_rows
# EVOLVE-BLOCK-END
