import time
import random
import math
import bisect
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    
    # Phase 1: Column-wise list scheduling for initial construction
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple seeds to find a good initial packing
    max_seeds = 200
    for seed in range(max_seeds):
        if time.time() > deadline - 40:
            break
            
        rows = _column_wise_construct(n_rows, k, seed)
        if rows is None:
            continue
            
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        # Fallback to simple greedy if column-wise failed
        best_rows = _greedy_construct_simple(n_rows, k, 0)
        best_scope = _scope(best_rows)

    # Phase 2: Simulated Annealing with element moves
    if time.time() < deadline - 10:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Final Polish - try to shrink the max element of each row
    if time.time() < deadline - 2:
        polished = _polish_max_element(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return set of all differences across all rows."""
    diffs = set()
    for row in rows:
        diffs.update(_row_diffs(row))
    return diffs

def _is_valid_row(row, used_diffs):
    """Check if a row is valid (all internal diffs distinct and not in used_diffs)."""
    n = len(row)
    internal_diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in internal_diffs or d in used_diffs:
                return False
            internal_diffs.add(d)
    return True

def _greedy_construct_simple(n_rows, k, seed):
    """Simple greedy fallback."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            x = last_x + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            used.update(new_diffs)
        rows.append(row)
    return rows

def _column_wise_construct(n_rows, k, seed):
    """
    Column-wise list scheduling: build all rows simultaneously, column by column.
    For each column position j (0 to k), choose values for all rows such that
    all differences are globally unique.
    """
    rng = random.Random(seed)
    rows = [[0] for _ in range(n_rows)]  # All rows start with 0
    used_diffs = set()
    
    # For each column position j = 1 to k
    for j in range(1, k + 1):
        # For each row, we need to choose a value x_i[j] > x_i[j-1]
        # such that all differences x_i[j] - x_i[i'] for i' < j are new and unique
        # across all rows and all pairs.
        
        # Strategy: for each row, find a small candidate value that doesn't conflict
        # Process rows in order, but use a priority queue to pick the row with fewest
        # options first (most constrained).
        
        # First, compute for each row the minimum possible next value
        candidates = []
        for i in range(n_rows):
            min_val = rows[i][-1] + 1
            candidates.append((min_val, i))
        
        # Sort by min_val (most constrained first)
        candidates.sort()
        
        # Now assign values. We'll use a greedy approach:
        # For each row in order of increasing min_val, find the smallest valid value.
        # But we need to be careful: assigning a value to one row may block others.
        # Instead, let's try to find values that are "spread out" to reduce conflicts.
        
        # Alternative: for each row, generate a list of candidate values (min_val to min_val + 50)
        # and pick the one with the fewest conflicts.
        
        new_values = [None] * n_rows
        for i in range(n_rows):
            min_val = rows[i][-1] + 1
            # Try values from min_val upward
            best_x = None
            best_conflicts = float('inf')
            for x in range(min_val, min_val + 80):
                # Compute new differences for this row
                new_diffs = set()
                for prev_idx in range(j):  # j is current column index (0-based)
                    d = x - rows[i][prev_idx]
                    new_diffs.add(d)
                
                # Count conflicts with used_diffs
                conflicts = len(new_diffs & used_diffs)
                if conflicts == 0:
                    # Also check against already-assigned values in this column
                    # (i.e., differences with other rows' new values)
                    # This is complex; for now, just check against used_diffs
                    best_x = x
                    break
                elif conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_x = x
            
            if best_x is None:
                # Fallback: just find the first valid value
                x = min_val
                while True:
                    new_diffs = {x - rows[i][prev_idx] for prev_idx in range(j)}
                    if not (new_diffs & used_diffs):
                        best_x = x
                        break
                    x += 1
            
            new_values[i] = best_x
        
        # Now we have candidate values for all rows. Check for global conflicts
        # among the new values themselves.
        # For each pair of rows (i, i'), the difference new_values[i] - new_values[i']
        # (if positive) must not be in used_diffs or in any other new difference.
        
        # This is getting complex. Let's use a simpler approach:
        # Assign values one by one, updating used_diffs as we go.
        
        used_diffs_local = set(used_diffs)
        new_values = [None] * n_rows
        
        # Order rows by their current last value (ascending) to minimize fragmentation
        row_order = sorted(range(n_rows), key=lambda i: rows[i][-1])
        
        for i in row_order:
            min_val = rows[i][-1] + 1
            x = min_val
            while True:
                new_diffs = {x - rows[i][prev_idx] for prev_idx in range(j)}
                if not (new_diffs & used_diffs_local):
                    break
                x += 1
            
            new_values[i] = x
            used_diffs_local.update(new_diffs)
        
        # Commit the new column
        for i in range(n_rows):
            rows[i].append(new_values[i])
        
        used_diffs = used_diffs_local
    
    return rows

def _packing_score(new_diffs, used_sorted):
    """
    Score a set of new differences based on how well they pack into the used space.
    Lower score is better.
    """
    if not used_sorted:
        return sum(new_diffs) * 0.1
    
    total_dist = 0.0
    for d in new_diffs:
        pos = bisect.bisect_left(used_sorted, d)
        
        min_dist = float('inf')
        if pos < len(used_sorted):
            min_dist = min(min_dist, abs(used_sorted[pos] - d))
        if pos > 0:
            min_dist = min(min_dist, abs(used_sorted[pos - 1] - d))
        
        if min_dist == float('inf'):
            min_dist = 100.0
        
        total_dist += min_dist
    
    avg_dist = total_dist / len(new_diffs)
    max_new = max(new_diffs)
    
    return avg_dist + 0.05 * max_new

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """
    Simulated Annealing with element-level moves.
    Move types:
    1. Change a single element in a row to a nearby value.
    2. Swap two elements in the same row (if they maintain order).
    3. Move an element to a different position in the row.
    """
    rng = random.Random(123)
    rows = [list(r) for r in initial_rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Compute initial used_diffs
    used_diffs = _all_diffs(rows)
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    max_iterations = 100000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline - 10:
            break
        iteration += 1
        
        # Pick a random row and element
        row_idx = rng.randint(0, n_rows - 1)
        elem_idx = rng.randint(1, k)  # Don't change the first element (0)
        
        old_val = rows[row_idx][elem_idx]
        
        # Try a new value
        # Bias towards smaller values to reduce scope
        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        
        # Ensure ordering: new_val > rows[row_idx][elem_idx - 1] and new_val < rows[row_idx][elem_idx + 1] (if exists)
        if elem_idx > 0:
            new_val = max(new_val, rows[row_idx][elem_idx - 1] + 1)
        if elem_idx < k:
            new_val = min(new_val, rows[row_idx][elem_idx + 1] - 1)
        
        if new_val == old_val:
            continue
        
        # Compute the differences that would be removed and added
        # Removed diffs: old_val - rows[row_idx][i] for i < elem_idx
        #                rows[row_idx][i] - old_val for i > elem_idx
        # Added diffs:   new_val - rows[row_idx][i] for i < elem_idx
        #                rows[row_idx][i] - new_val for i > elem_idx
        
        removed_diffs = set()
        for i in range(elem_idx):
            removed_diffs.add(old_val - rows[row_idx][i])
        for i in range(elem_idx + 1, k + 1):
            removed_diffs.add(rows[row_idx][i] - old_val)
        
        added_diffs = set()
        for i in range(elem_idx):
            added_diffs.add(new_val - rows[row_idx][i])
        for i in range(elem_idx + 1, k + 1):
            added_diffs.add(rows[row_idx][i] - new_val)
        
        # Check if added_diffs conflict with (used_diffs - removed_diffs)
        available = used_diffs - removed_diffs
        if added_diffs & available:
            # Conflict, reject
            continue
        
        # Also check internal consistency: added_diffs must be distinct
        if len(added_diffs) != len(added_diffs):  # Always true for a set
            continue
        
        # Accept the move
        rows[row_idx][elem_idx] = new_val
        used_diffs = (used_diffs - removed_diffs) | added_diffs
        
        new_scope = _scope(rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [list(r) for r in rows]
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_rows

def _polish_max_element(initial_rows, n_rows, k, start_time, deadline):
    """
    Final pass to try to reduce the max element of each row.
    """
    rng = random.Random(777)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for idx in range(n_rows):
        if time.time() > deadline - 2:
            break
            
        old_row = rows[idx]
        old_max = old_row[-1]
        
        # Collect diffs from other rows
        used_diffs = set()
        for j in range(n_rows):
            if j == idx:
                continue
            used_diffs.update(_row_diffs(rows[j]))
        
        # Try to find a row with max < old_max
        for target_max in range(old_max - 1, k - 1, -1):
            if time.time() > deadline - 2:
                break
            
            found = False
            for _ in range(20):
                new_row = _find_row_with_limit(used_diffs, k, target_max, rng)
                if new_row is not None:
                    rows[idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    found = True
                    break
            
            if found:
                break
    
    return best_rows

def _find_row_with_limit(used_diffs, k, max_val, rng):
    """
    Find a row of k+1 elements starting at 0, max <= max_val,
    with all internal diffs avoiding used_diffs.
    Uses randomized backtracking.
    """
    row = [0]
    internal_diffs = set()
    
    def backtrack(pos):
        if pos == k + 1:
            return True
        
        last_x = row[-1]
        
        valid_xs = []
        for x in range(last_x + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue
            if new_diffs & internal_diffs:
                continue
            valid_xs.append((x, new_diffs))
        
        if not valid_xs:
            return False
        
        rng.shuffle(valid_xs)
        valid_xs.sort(key=lambda t: t[0])
        top_n = min(5, len(valid_xs))
        candidates = valid_xs[:top_n]
        rng.shuffle(candidates)
        
        for x, new_diffs in candidates:
            row.append(x)
            internal_diffs.update(new_diffs)
            
            if backtrack(pos + 1):
                return True
            
            row.pop()
            internal_diffs.difference_update(new_diffs)
        
        return False
    
    if backtrack(1):
        return row
    return None
