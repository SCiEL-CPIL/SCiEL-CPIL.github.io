"""Difference triangle set (4,7) – simulated-annealing construction."""
import time
import random
from itertools import combinations

def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)

def _all_diffs(rows):
    """Set of all positive differences within each row."""
    s = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                s.add(r[j] - r[i])
    return s

def _is_valid(rows):
    """Check that all positive differences are distinct across all rows."""
    total = 0
    for r in rows:
        total += len(r) * (len(r) - 1) // 2
    return len(_all_diffs(rows)) == total

def _greedy_row(used, k, max_val):
    """Build one row greedily given the set of used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_val:
            return None
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row

def _greedy_construct(n_rows, k, max_val=200):
    """Greedy construction for n_rows rows of k+1 entries."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, max_val)
        if row is None:
            return None
        rows.append(row)
    return rows

def _random_row(k, max_val, used):
    """Generate a random valid row given used differences."""
    row = [0]
    x = 0
    attempts = 0
    while len(row) < k + 1:
        if attempts > 10000:
            return None
        x += 1
        if x > max_val:
            return None
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
        attempts += 1
    return row

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope), such that all positive
    differences within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Initial greedy solution
    best_rows = _greedy_construct(n_rows, k, max_val=250)
    if best_rows is None:
        # Fallback: very loose greedy
        best_rows = _greedy_construct(n_rows, k, max_val=500)
        if best_rows is None:
            # Last resort: return something (may be invalid but won't crash)
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                rows.append(row)
            return rows

    best_scope = _scope(best_rows)
    best_valid = _is_valid(best_rows)

    # Simulated annealing to reduce scope
    rng = random.Random(42)
    
    # We'll work with a representation where each row is a list of k+1 ints.
    # Move: pick a row, pick a position (1..k), change the value.
    
    temperature = best_scope * 0.5
    cooling = 0.9999
    min_temp = 1e-6
    
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    while temperature > min_temp:
        if time.time() > deadline:
            break
        
        # Pick a random row and a random position to modify
        row_idx = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)  # positions 1..k (index 0 is always 0)
        
        old_val = current_rows[row_idx][pos]
        
        # Try to move the value: decrease, increase, or random nearby
        move_type = rng.random()
        if move_type < 0.4:
            # Try to decrease (toward smaller scope)
            delta = rng.randint(1, max(1, old_val // 5))
            new_val = max(pos, old_val - delta)  # must stay increasing
        elif move_type < 0.7:
            # Try to increase slightly
            delta = rng.randint(1, max(1, old_val // 10))
            new_val = old_val + delta
        else:
            # Random nearby
            delta = rng.randint(-max(1, old_val // 4), max(1, old_val // 4))
            new_val = max(pos, old_val + delta)
        
        # Check if the new value maintains strict increase within the row
        if pos > 0 and new_val <= current_rows[row_idx][pos - 1]:
            temperature *= cooling
            continue
        if pos < k and new_val >= current_rows[row_idx][pos + 1]:
            temperature *= cooling
            continue
        
        # Apply the move temporarily
        current_rows[row_idx][pos] = new_val
        new_scope = _scope(current_rows)
        
        # Check validity
        new_valid = _is_valid(current_rows)
        
        if new_valid and new_scope < current_scope:
            # Accept (improvement)
            current_scope = new_scope
            if current_scope < best_scope and new_valid:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
                best_valid = True
        elif new_valid and new_scope == current_scope:
            # Accept sideways
            pass
        else:
            # Compute delta in "cost"
            # Cost = scope + large penalty if invalid
            old_cost = current_scope + (10000 if not best_valid else 0)
            # Actually, we need to track validity of current state
            # Let's redefine: we only keep valid states in current_rows
            # So if new state is invalid, we reject unless we allow invalid states in SA
            # For simplicity, reject invalid states
            current_rows[row_idx][pos] = old_val
            temperature *= cooling
            continue
        
        temperature *= cooling
    
    # If best is valid, return it
    if best_valid:
        return best_rows
    
    # Otherwise return the greedy solution (which should be valid)
    return best_rows
