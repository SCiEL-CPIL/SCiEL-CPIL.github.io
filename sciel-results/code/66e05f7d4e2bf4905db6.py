"""
Difference triangle set (7,5): construct 7 rows of 6 increasing integers starting at 0
so that all positive differences within rows are distinct across all rows.
Minimize the largest entry.

Approach: Simulated Annealing with a hybrid move set.
1. Randomized greedy construction to get an initial feasible solution.
2. SA with two types of moves:
   a) Element-level: pick a row, pick an element (not the first 0), try to move it to a nearby value.
   b) Row-level: pick a row, try to replace it entirely with a new greedy row (given used diffs).
3. Deadline guard to ensure we return within time limit.
"""

import time
import random
import sys

def _construct_greedy_row(used, k, start_val=0):
    """
    Build a single row greedily: start at 0, then pick smallest next mark
    such that all new differences are unused.
    Returns the row (list of k+1 ints) or None if impossible.
    """
    row = [0]
    x = 1
    while len(row) < k + 1:
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
            x += 1
        else:
            x += 1
    return row

def _construct_greedy_row_with_max(used, k, max_val):
    """
    Build a single row greedily but stop if we exceed max_val.
    Returns the row or None.
    """
    row = [0]
    x = 1
    while len(row) < k + 1:
        if x > max_val:
            return None
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
            x += 1
        else:
            x += 1
    return row

def _randomized_greedy_row(used, k, max_val, rng, bias=1.0):
    """
    Build a row with some randomization: sometimes skip a valid value to explore.
    """
    row = [0]
    x = 1
    attempts = 0
    max_attempts = max_val * 3
    while len(row) < k + 1:
        if x > max_val or attempts > max_attempts:
            return None
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            # With probability bias, take it; otherwise skip
            if rng.random() < bias:
                row.append(x)
                used |= new_diffs
                x += 1
            else:
                x += 1
        else:
            x += 1
        attempts += 1
    return row

def _get_used_diffs(rows):
    """Compute the set of all positive differences used in the given rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used.add(row[j] - row[i])
    return used

def _row_max(row):
    return row[-1]

def _total_max(rows):
    return max(_row_max(r) for r in rows)

def construct(n_rows, k):
    """
    Main entry point. Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    rng = random.Random(42)
    
    # Phase 1: Initial construction via greedy
    used = set()
    rows = []
    for i in range(n_rows):
        row = _construct_greedy_row(used, k)
        if row is None:
            # Fallback: just use a wide range
            row = list(range(0, (k+1) * 20, 20))
            used = _get_used_diffs([row])
        else:
            rows.append(row)
    
    if len(rows) < n_rows:
        # Shouldn't happen, but safety
        while len(rows) < n_rows:
            rows.append(list(range(0, (k+1) * 50, 50)))
    
    best_rows = [r[:] for r in rows]
    best_max = _total_max(best_rows)
    
    # Phase 2: Simulated Annealing
    # State: rows (list of lists)
    # Cost: total_max = max over all rows of row[-1]
    
    current_rows = [r[:] for r in rows]
    current_max = _total_max(current_rows)
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    cooling = 0.9995
    
    # We'll track used diffs for quick checking
    used_set = _get_used_diffs(current_rows)
    
    step = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_steps > max_no_improve:
            # Restart with a fresh greedy construction
            used = set()
            new_rows = []
            for i in range(n_rows):
                row = _randomized_greedy_row(used, k, current_max + 10, rng, bias=0.8)
                if row is None:
                    row = _construct_greedy_row(used, k)
                if row is not None:
                    new_rows.append(row)
            if len(new_rows) == n_rows:
                new_max = _total_max(new_rows)
                if new_max < current_max:
                    current_rows = new_rows
                    current_max = new_max
                    used_set = _get_used_diffs(current_rows)
                no_improve_steps = 0
            else:
                break
        
        # Choose a move type
        if rng.random() < 0.7:
            # Element-level move: pick a row, pick an element (index >= 1), try new value
            row_idx = rng.randrange(n_rows)
            row = current_rows[row_idx]
            elem_idx = rng.randrange(1, len(row))  # skip index 0 (always 0)
            
            # Try a nearby value
            delta = rng.randint(-5, 5)
            new_val = row[elem_idx] + delta
            
            # Check constraints: must be > row[elem_idx-1] and < row[elem_idx+1] (if exists)
            if new_val <= row[elem_idx-1]:
                continue
            if elem_idx < len(row) - 1 and new_val >= row[elem_idx+1]:
                continue
            
            # Compute differences that would be removed and added
            old_val = row[elem_idx]
            removed_diffs = set()
            added_diffs = set()
            
            for j in range(len(row)):
                if j == elem_idx:
                    continue
                if j < elem_idx:
                    # row[elem_idx] - row[j]
                    removed_diffs.add(old_val - row[j])
                    added_diffs.add(new_val - row[j])
                else:
                    # row[j] - row[elem_idx]
                    removed_diffs.add(row[j] - old_val)
                    added_diffs.add(row[j] - new_val)
            
            # Check if added diffs conflict with used_set (excluding removed ones)
            effective_used = used_set - removed_diffs
            if added_diffs & effective_used:
                continue
            
            # Apply move
            row[elem_idx] = new_val
            used_set = effective_used | added_diffs
            
            new_max = _total_max(current_rows)
            delta_cost = new_max - current_max
            
            if delta_cost <= 0 or rng.random() < min(1.0, exp_neg(delta_cost, T_start * (cooling ** step))):
                current_max = new_max
                if new_max < best_max:
                    best_rows = [r[:] for r in current_rows]
                    best_max = new_max
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Revert
                row[elem_idx] = old_val
                used_set = used_set - added_diffs | removed_diffs
                no_improve_steps += 1
        else:
            # Row-level move: replace a row with a new greedy row
            row_idx = rng.randrange(n_rows)
            old_row = current_rows[row_idx]
            
            # Remove old row's diffs
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i+1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            temp_used = used_set - old_diffs
            
            # Try to build a new row with max_val = current_max (or slightly less)
            target_max = current_max
            if rng.random() < 0.5:
                target_max = max(1, current_max - rng.randint(0, 3))
            
            new_row = _randomized_greedy_row(temp_used, k, target_max, rng, bias=0.9)
            
            if new_row is not None:
                new_row_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        new_row_diffs.add(new_row[j] - new_row[i])
                
                # Check no conflicts
                if not (new_row_diffs & temp_used):
                    current_rows[row_idx] = new_row
                    used_set = temp_used | new_row_diffs
                    
                    new_max = _total_max(current_rows)
                    delta_cost = new_max - current_max
                    
                    if delta_cost <= 0 or rng.random() < min(1.0, exp_neg(delta_cost, T_start * (cooling ** step))):
                        current_max = new_max
                        if new_max < best_max:
                            best_rows = [r[:] for r in current_rows]
                            best_max = new_max
                            no_improve_steps = 0
                        else:
                            no_improve_steps += 1
                    else:
                        # Revert
                        current_rows[row_idx] = old_row
                        used_set = used_set - new_row_diffs | old_diffs
                        no_improve_steps += 1
            else:
                no_improve_steps += 1
        
        step += 1
    
    # Phase 3: Final tightening - try to reduce each row's max
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        improved = False
        for row_idx in range(n_rows):
            row = best_rows[row_idx]
            current_row_max = row[-1]
            
            # Try to find a smaller row
            used_others = set()
            for r_idx in range(n_rows):
                if r_idx == row_idx:
                    continue
                r = best_rows[r_idx]
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        used_others.add(r[j] - r[i])
            
            # Try greedy with decreasing max
            for trial_max in range(current_row_max - 1, 0, -1):
                new_row = _construct_greedy_row_with_max(used_others, k, trial_max)
                if new_row is not None:
                    best_rows[row_idx] = new_row
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_rows

def exp_neg(x, T):
    """Compute exp(-x/T) safely."""
    if T <= 0:
        return 0.0
    if x / T > 50:
        return 0.0
    import math
    return math.exp(-x / T)
