import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Get an initial valid solution using greedy
    rows = _greedy(n_rows, k)
    best_scope = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    
    rng = random.Random(12345)
    
    # Try to improve by local search
    while time.time() < deadline:
        # Randomly perturb the current best solution
        candidate = [list(r) for r in best_rows]
        
        # Try to reduce the maximum value by adjusting entries
        improved = False
        for _ in range(100):  # Limit iterations per attempt
            if time.time() > deadline - 1.0:
                break
            
            # Pick a random row and a random entry (not the first)
            row_idx = rng.randint(0, n_rows - 1)
            entry_idx = rng.randint(1, k)  # Skip the 0
            
            # Try to decrease this entry
            old_val = candidate[row_idx][entry_idx]
            
            # Try values from old_val-1 down to candidate[row_idx][entry_idx-1]+1
            for new_val in range(old_val - 1, candidate[row_idx][entry_idx-1], -1):
                # Check if this creates a conflict
                if _is_valid_change(candidate, row_idx, entry_idx, new_val):
                    candidate[row_idx][entry_idx] = new_val
                    improved = True
                    break
            
            if not improved:
                break
        
        if improved:
            new_scope = max(max(r) for r in candidate)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = candidate
                # Reset perturbation counter
                continue
        
        # If no improvement, try a random restart with slight modification
        # Modify a random entry to a random valid value
        row_idx = rng.randint(0, n_rows - 1)
        entry_idx = rng.randint(1, k)
        min_val = candidate[row_idx][entry_idx-1] + 1
        max_val = best_scope
        if max_val > min_val:
            new_val = rng.randint(min_val, max_val)
            if _is_valid_change(candidate, row_idx, entry_idx, new_val):
                candidate[row_idx][entry_idx] = new_val
                new_scope = max(max(r) for r in candidate)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = candidate
    
    return best_rows


def _greedy(n_rows, k):
    """Simple greedy to get a valid starting point."""
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


def _is_valid_change(rows, row_idx, entry_idx, new_val):
    """
    Check if changing rows[row_idx][entry_idx] to new_val maintains validity.
    The new value must be > previous entry and < next entry (if exists).
    All differences must remain distinct across all rows.
    """
    row = rows[row_idx]
    
    # Check ordering
    if entry_idx > 0 and new_val <= row[entry_idx - 1]:
        return False
    if entry_idx < len(row) - 1 and new_val >= row[entry_idx + 1]:
        return False
    
    # Collect all differences that will be affected
    # Differences involving entry_idx in row_idx
    old_val = row[entry_idx]
    
    # Differences to remove: (old_val - row[i]) for i < entry_idx
    # and (row[j] - old_val) for j > entry_idx
    # Differences to add: (new_val - row[i]) for i < entry_idx
    # and (row[j] - new_val) for j > entry_idx
    
    # First, collect all differences in the entire set
    all_diffs = {}
    for r_idx in range(len(rows)):
        r = rows[r_idx]
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    # Already a duplicate - but we assume current state is valid
                    pass
                all_diffs[d] = (r_idx, i, j)
    
    # Remove old differences involving (row_idx, entry_idx)
    for i in range(entry_idx):
        d = old_val - row[i]
        if d in all_diffs and all_diffs[d] == (row_idx, i, entry_idx):
            del all_diffs[d]
    
    for j in range(entry_idx + 1, len(row)):
        d = row[j] - old_val
        if d in all_diffs and all_diffs[d] == (row_idx, entry_idx, j):
            del all_diffs[d]
    
    # Add new differences
    for i in range(entry_idx):
        d = new_val - row[i]
        if d in all_diffs:
            return False
        all_diffs[d] = (row_idx, i, entry_idx)
    
    for j in range(entry_idx + 1, len(row)):
        d = row[j] - new_val
        if d in all_diffs:
            return False
        all_diffs[d] = (row_idx, entry_idx, j)
    
    return True
