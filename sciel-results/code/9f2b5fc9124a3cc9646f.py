"""Rewritten program for difference triangle set (7,5)."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry."""
    
    deadline = time.time() + 115  # safety margin before 150s limit
    
    # Phase 1: Greedy construction with column-wise ordering
    best_rows = _column_wise_greedy(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    # Phase 2: Simulated annealing to improve
    if time.time() < deadline - 5:
        improved = _simulated_annealing(n_rows, k, best_rows, best_max, deadline)
        if improved is not None:
            imp_max = max(max(row) for row in improved)
            if imp_max < best_max:
                best_rows = improved
                best_max = imp_max
    
    # Phase 3: Local search / reduction attempts
    if time.time() < deadline - 2:
        reduced = _reduce_max(n_rows, k, best_rows, deadline)
        if reduced is not None:
            red_max = max(max(row) for row in reduced)
            if red_max < best_max:
                best_rows = reduced
                best_max = red_max
    
    return best_rows


def _row_diffs(row):
    """Get all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Get all differences across all rows."""
    all_d = set()
    for row in rows:
        all_d |= _row_diffs(row)
    return all_d


def _is_valid(rows):
    """Check if all differences are distinct across rows."""
    all_d = _all_diffs(rows)
    total = sum(len(row) * (len(row) - 1) // 2 for row in rows)
    return len(all_d) == total


def _scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(row) for row in rows)


def _column_wise_greedy(n_rows, k):
    """Column-wise greedy: build each row position by position,
    choosing the smallest value that doesn't conflict with existing differences."""
    
    used_diffs = set()
    rows = []
    
    for r in range(n_rows):
        row = [0]
        # Build positions 1 through k
        for pos in range(1, k + 1):
            # Try values in increasing order from last+1
            start = row[-1] + 1
            found = False
            for val in range(start, 500):
                # Check new differences
                new_diffs = set()
                valid = True
                for a in row:
                    d = val - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(val)
                    used_diffs |= new_diffs
                    found = True
                    break
            if not found:
                # Fallback: just pick a large value
                row.append(row[-1] + 100)
                new_diffs = {row[-1] - a for a in row[:-1]}
                used_diffs |= new_diffs
        rows.append(row)
    
    return rows


def _packing_score(rows, used_diffs):
    """Score how tightly packed the differences are.
    Lower is better: prefer rows where differences are smaller and more clustered."""
    score = 0.0
    for row in rows:
        diffs = _row_diffs(row)
        if diffs:
            max_d = max(diffs)
            min_d = min(diffs)
            # Penalize large spread
            score += max_d * 0.1 + (max_d - min_d) * 0.05
    return score


def _simulated_annealing(n_rows, k, initial_rows, initial_max, deadline):
    """Simulated annealing to reduce the maximum entry."""
    
    rows = [list(r) for r in initial_rows]
    used_diffs = _all_diffs(rows)
    current_max = max(max(row) for row in rows)
    
    best_rows = [list(r) for r in rows]
    best_max = current_max
    
    temperature = 50.0
    cooling = 0.995
    min_temp = 0.1
    
    start_time = time.time()
    
    while time.time() < deadline - 5 and temperature > min_temp:
        # Pick a random row and a random position to modify
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)  # positions 1..k (not 0)
        
        old_val = rows[r][pos]
        
        # Try a new value: slightly smaller or larger
        delta = random.choice([-3, -2, -1, 1, 2, 3, 5, 8, -5, -8])
        new_val = old_val + delta
        
        # Must maintain strict increase
        if pos > 0 and new_val <= rows[r][pos-1]:
            continue
        if pos < k and new_val >= rows[r][pos+1]:
            continue
        if new_val <= 0:
            continue
        
        # Compute old differences for this row
        old_row_diffs = _row_diffs(rows[r])
        
        # Compute new differences for this row
        rows[r][pos] = new_val
        new_row_diffs = _row_diffs(rows[r])
        
        # Check for conflicts
        removed = old_row_diffs - new_row_diffs
        added = new_row_diffs - old_row_diffs
        conflicts = added & (used_diffs - removed)
        
        if conflicts:
            rows[r][pos] = old_val
            continue
        
        # Accept with probability based on temperature
        old_max = current_max
        new_max = max(max(row) for row in rows)
        
        if new_max <= current_max:
            # Always accept improvements
            current_max = new_max
            used_diffs = (used_diffs - old_row_diffs) | new_row_diffs
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(row) for row in rows]
        else:
            # Accept with probability exp(-(new_max - old_max) / temperature)
            prob = math.exp(-(new_max - old_max) / temperature)
            if random.random() < prob:
                current_max = new_max
                used_diffs = (used_diffs - old_row_diffs) | new_row_diffs
            else:
                rows[r][pos] = old_val
        
        temperature *= cooling
    
    if best_max < initial_max:
        return best_rows
    return None


def _reduce_max(n_rows, k, rows, deadline):
    """Try to reduce the maximum entry by adjusting the row with the largest value."""
    
    rows = [list(r) for r in rows]
    used_diffs = _all_diffs(rows)
    current_max = max(max(row) for row in rows)
    
    # Find the row with the maximum entry
    max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
    max_val = max(rows[max_row_idx])
    
    # Try to reduce the last element of that row
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        target_pos = k  # last position
        old_val = rows[max_row_idx][target_pos]
        
        # Try reducing by 1
        new_val = old_val - 1
        if new_val <= rows[max_row_idx][target_pos - 1]:
            break
        
        old_row_diffs = _row_diffs(rows[max_row_idx])
        rows[max_row_idx][target_pos] = new_val
        new_row_diffs = _row_diffs(rows[max_row_idx])
        
        added = new_row_diffs - old_row_diffs
        removed = old_row_diffs - new_row_diffs
        conflicts = added & (used_diffs - removed)
        
        if not conflicts:
            used_diffs = (used_diffs - old_row_diffs) | new_row_diffs
            new_max = max(max(row) for row in rows)
            if new_max < current_max:
                current_max = new_max
                # Continue trying to reduce further
                continue
            else:
                rows[max_row_idx][target_pos] = old_val
                break
        else:
            rows[max_row_idx][target_pos] = old_val
            break
    
    new_max = max(max(row) for row in rows)
    if new_max < max(max(row) for row in rows):
        return rows
    return None
# EVOLVE-BLOCK-END
