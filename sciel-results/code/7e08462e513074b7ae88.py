import time
import random
import sys
from itertools import combinations

def _validate(rows):
    """Check that all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    return max(max(row) for row in rows)

def _greedy_row(used, k, start=0):
    """Build a single row greedily: pick smallest next value with no conflict."""
    row = [0]
    x = start
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used |= new_diffs
    return row

def _greedy_full(n_rows, k):
    """Standard greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows

def _randomized_greedy(used, k, start=0, jitter=0):
    """Build a row with some randomization in choice of next element."""
    row = [0]
    x = start
    attempts = 0
    max_attempts = 5000
    while len(row) < k + 1 and attempts < max_attempts:
        # Try a small window of candidates
        candidates = []
        base_x = x
        for dx in range(0, 1 + jitter):
            cand = base_x + dx
            new_diffs = {cand - a for a in row}
            if not (new_diffs & used):
                candidates.append(cand)
        if not candidates:
            x += 1
            attempts += 1
            continue
        if jitter > 0 and len(candidates) > 1:
            # Pick a random candidate, biased toward smaller
            idx = min(len(candidates) - 1, random.randint(0, max(0, len(candidates) // 2)))
            x = candidates[idx]
        else:
            x = candidates[0]
        row.append(x)
        used |= {x - a for a in row[:-1]}
        attempts += 1
    if len(row) < k + 1:
        return None
    return row

def _local_search(rows, k, max_iters=20000, deadline=None):
    """
    Local search: try to reduce the scope by moving elements.
    Each row has k+1 elements starting at 0. We try to reduce the max element.
    """
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    n = len(rows)
    
    def check_validity(rows_):
        used = set()
        for row in rows_:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def row_diffs(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Precompute differences for each row
    row_diff_sets = [row_diffs(r) for r in best_rows]
    all_diffs = set()
    for rs in row_diff_sets:
        all_diffs |= rs
    
    def is_valid_change(row_idx, old_val, new_val):
        """Check if changing row[row_idx][pos] from old_val to new_val keeps validity."""
        row = best_rows[row_idx]
        # Check increasing
        pos = None
        for i, v in enumerate(row):
            if v == old_val:
                pos = i
                break
        if pos is None:
            return False
        if pos > 0 and new_val <= row[pos-1]:
            return False
        if pos < len(row)-1 and new_val >= row[pos+1]:
            return False
        if new_val == old_val:
            return True
        
        # Compute new diffs for this row
        new_row = list(row)
        new_row[pos] = new_val
        new_diffs = row_diffs(new_row)
        
        # Check no overlap with other rows
        for other_idx in range(n):
            if other_idx == row_idx:
                continue
            if new_diffs & row_diff_sets[other_idx]:
                return False
        return True
    
    iters = 0
    while iters < max_iters:
        if deadline and time.time() > deadline:
            break
        iters += 1
        
        # Pick a random row and a random non-zero element to move
        row_idx = random.randint(0, n - 1)
        row = best_rows[row_idx]
        # Only consider elements that can be reduced (not the last one necessarily)
        # Try to reduce the scope: pick the last element or any element
        pos = random.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Try to move to a smaller value
        # Search for a valid new value
        found = False
        for new_val in range(row[pos-1] + 1, old_val):
            if is_valid_change(row_idx, old_val, new_val):
                # Apply change
                best_rows[row_idx][pos] = new_val
                row_diff_sets[row_idx] = row_diffs(best_rows[row_idx])
                new_scope = _scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                found = True
                break
        
        if not found:
            # Try a random perturbation to escape local minima
            # Move an element to a random nearby position
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            # Try a few random nearby values
            for _ in range(10):
                delta = random.choice([-1, 1, -2, 2, -3, 3, -5, 5])
                new_val = old_val + delta
                if new_val <= row[pos-1] or (pos < len(row)-1 and new_val >= row[pos+1]):
                    continue
                if new_val <= 0:
                    continue
                if is_valid_change(row_idx, old_val, new_val):
                    best_rows[row_idx][pos] = new_val
                    row_diff_sets[row_idx] = row_diffs(best_rows[row_idx])
                    new_scope = _scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    break
    
    return best_rows, best_scope

def _backtracking_row(used, k, current_row, x_start, deadline=None):
    """
    Backtracking to build a single row given used differences.
    current_row: list of elements so far (starts with [0])
    x_start: minimum value to try for next element
    Returns (row, used_new) or (None, None)
    """
    if deadline and time.time() > deadline:
        return None, None
    
    if len(current_row) == k + 1:
        return list(current_row), set(used)
    
    x = x_start
    max_x = x_start + 200  # Limit search depth per level
    
    while x <= max_x:
        if deadline and time.time() > deadline:
            return None, None
        
        new_diffs = {x - a for a in current_row}
        if not (new_diffs & used):
            new_used = used | new_diffs
            current_row.append(x)
            result, new_used_result = _backtracking_row(new_used, k, current_row, x + 1, deadline)
            current_row.pop()
            if result is not None:
                return result, new_used_result
        x += 1
    
    return None, None

def _construct_with_backtracking(n_rows, k, deadline):
    """Construct rows using backtracking for each row."""
    used = set()
    rows = []
    
    for r in range(n_rows):
        if deadline and time.time() > deadline:
            break
        result, new_used = _backtracking_row(used, k, [0], 1, deadline)
        if result is None:
            # Fall back to greedy for this row
            result = _greedy_row(used, k)
            new_used = used | {result[j] - result[i] for i in range(len(result)) for j in range(i+1, len(result))}
        rows.append(result)
        used = new_used
    
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Phase 1: Initial greedy construction
    rows = _greedy_full(n_rows, k)
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    # Phase 2: Try backtracking-based construction (may find better first row)
    if time.time() < deadline - 60:
        bt_rows = _construct_with_backtracking(n_rows, k, deadline - 30)
        if len(bt_rows) == n_rows:
            bt_scope = _scope(bt_rows)
            if bt_scope < best_scope:
                best_scope = bt_scope
                best_rows = [list(r) for r in bt_rows]
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 10:
        improved_rows, improved_scope = _local_search(
            best_rows, k,
            max_iters=50000,
            deadline=deadline - 10
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Try randomized restarts if time permits
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            random.seed(restart * 12345 + 67890)
            used = set()
            rand_rows = []
            success = True
            for r in range(n_rows):
                if time.time() > deadline - 10:
                    success = False
                    break
                jitter = random.randint(0, 3)
                row = _randomized_greedy(used, k, jitter=jitter)
                if row is None:
                    success = False
                    break
                rand_rows.append(row)
                used |= {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
            
            if success and len(rand_rows) == n_rows:
                rand_scope = _scope(rand_rows)
                if rand_scope < best_scope:
                    best_scope = rand_scope
                    best_rows = [list(r) for r in rand_rows]
                
                # Quick local search on this
                if time.time() < deadline - 5:
                    ls_rows, ls_scope = _local_search(
                        best_rows, k,
                        max_iters=10000,
                        deadline=deadline - 5
                    )
                    if ls_scope < best_scope:
                        best_scope = ls_scope
                        best_rows = ls_rows
    
    # Final validation
    if not _validate(best_rows):
        # Fallback to pure greedy
        best_rows = _greedy_full(n_rows, k)
    
    return best_rows
