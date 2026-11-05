"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _greedy_construct(n_rows, k, used=None):
    """
    Greedy construction: build rows one by one, choosing the smallest next mark
    such that all new differences are unused.
    """
    if used is None:
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
    return rows, used

def _compute_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """Check that all rows start at 0, are strictly increasing, have k+1 elements,
    and all positive differences are distinct across all rows."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def _try_improve_row(rows, row_idx, k, used, max_iter=200):
    """
    Try to reduce the last element of a specific row by local search.
    We attempt to find an alternative row with the same set of differences
    but a smaller maximum element, or to restructure the row.
    """
    best_row = list(rows[row_idx])
    best_scope = max(best_row)
    
    # Get the differences used by this row
    row_diffs = set()
    for i in range(len(best_row)):
        for j in range(i + 1, len(best_row)):
            row_diffs.add(best_row[j] - best_row[i])
    
    other_diffs = used - row_diffs
    
    # Try to find a better row with the same differences but smaller max
    # This is hard in general, so instead try local moves
    # Try swapping elements or shifting
    
    # Simple approach: try to reduce the last element by 1, 2, ... 
    # and check if we can fill in the remaining positions
    for target_last in range(best_scope - 1, 0, -1):
        if target_last >= best_scope:
            continue
        # Try to construct a row ending at target_last with the same differences
        # This is complex; skip for now and use a simpler heuristic
        
        # Instead, try a local perturbation: pick a random element (not first)
        # and try to decrease it by 1, checking validity
        success = False
        for _ in range(max_iter):
            new_row = list(best_row)
            idx = random.randint(1, len(new_row) - 1)
            if new_row[idx] <= 1:
                continue
            new_row[idx] -= 1
            # Check if still strictly increasing
            if any(new_row[i+1] <= new_row[i] for i in range(len(new_row)-1)):
                continue
            # Check differences
            new_diffs = set()
            valid = True
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            if not valid:
                continue
            # Check against other rows' differences
            if new_diffs & other_diffs:
                continue
            # Valid! Update
            if max(new_row) < best_scope:
                best_row = new_row
                best_scope = max(best_row)
                row_diffs = new_diffs
                success = True
            # Continue trying to improve further
    
    return best_row, row_diffs, success

def _local_search(rows, k, used, time_limit):
    """
    Perform local search to reduce the overall scope.
    """
    start_time = time.time()
    best_scope = _compute_scope(rows)
    best_rows = [list(r) for r in rows]
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # Pick a random row to try to improve
        row_idx = random.randint(0, len(rows) - 1)
        new_row, new_diffs, improved = _try_improve_row(
            best_rows, row_idx, k, used, max_iter=50
        )
        
        if improved and max(new_row) < max(best_rows[row_idx]):
            # Update used set
            old_diffs = set()
            for i in range(len(best_rows[row_idx])):
                for j in range(i+1, len(best_rows[row_idx])):
                    old_diffs.add(best_rows[row_idx][j] - best_rows[row_idx][ i])
            
            used -= old_diffs
            used |= new_diffs
            best_rows[row_idx] = new_row
            no_improve_count = 0
            
            new_scope = _compute_scope(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
        else:
            no_improve_count += 1
        
        if no_improve_count > 200:
            break
    
    return best_rows, best_scope

def _column_wise_construction(n_rows, k, time_limit):
    """
    Alternative construction: build columns (positions) rather than rows.
    This may allow global constraint awareness earlier.
    """
    start_time = time.time()
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    for col in range(1, k + 1):
        # For each column, assign values to all rows
        # Sort rows by current max to prioritize constrained rows
        row_order = sorted(range(n_rows), key=lambda i: max(rows[i]))
        
        assigned = [False] * n_rows
        candidates = {}
        
        for i in row_order:
            # Find smallest value > rows[i][-1] that doesn't conflict
            x = rows[i][-1] + 1
            while True:
                new_diffs = {x - a for a in rows[i]}
                if not new_diffs & used:
                    candidates[i] = x
                    break
                x += 1
        
        # Assign in order, updating used
        for i in row_order:
            rows[i].append(candidates[i])
            new_diffs = {candidates[i] - a for a in rows[i][:-1]}
            used |= new_diffs
        
        if time.time() - start_time > time_limit:
            break
    
    # If we didn't complete all columns, fall back to greedy completion
    if any(len(row) < k + 1 for row in rows):
        for i in range(n_rows):
            while len(rows[i]) < k + 1:
                x = rows[i][-1] + 1
                while True:
                    new_diffs = {x - a for a in rows[i]}
                    if not new_diffs & used:
                        rows[i].append(x)
                        used |= new_diffs
                        break
                    x += 1
    
    return rows, used

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    are distinct across all rows.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Phase 1: Initial construction using column-wise approach
    rows, used = _column_wise_construction(n_rows, k, time_limit=30)
    if time.time() > deadline - 10:
        return rows
    
    # Verify validity
    if not _is_valid(rows, k):
        # Fallback to greedy
        rows, used = _greedy_construct(n_rows, k)
    
    best_scope = _compute_scope(rows)
    best_rows = [list(r) for r in rows]
    
    # Phase 2: Local search to improve
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        improved_rows, improved_scope = _local_search(
            best_rows, k, used, time_limit=remaining_time
        )
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
    
    # Phase 3: Try a few more greedy constructions with different orderings
    if time.time() < deadline - 15:
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            # Shuffle row order for greedy
            shuffled_rows, shuffled_used = _greedy_construct(n_rows, k)
            scope = _compute_scope(shuffled_rows)
            if scope < best_scope:
                best_rows = shuffled_rows
                best_scope = scope
    
    return best_rows

# EVOLVE-BLOCK-END
