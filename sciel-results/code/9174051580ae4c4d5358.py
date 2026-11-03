import time
import random
import sys
import math

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for d in _compute_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, skip_prob=0.0, seed=None):
    """
    Greedy initialization with optional random skipping to create diversity.
    """
    if seed is not None:
        random.seed(seed)
    
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # With some probability, skip to create diversity
                if skip_prob > 0 and random.random() < skip_prob:
                    # Skip a few valid values
                    skipped = False
                    for _ in range(random.randint(1, 3)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        # If we couldn't skip, just take current x if valid
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def _rebuild_row_greedy(forbidden, k, max_scope_hint):
    """
    Rebuild a row using greedy approach to minimize its maximum element.
    forbidden: set of differences already used by other rows.
    Returns the best row found or None.
    """
    row = [0]
    local_used = set()
    
    for _ in range(k):
        found = False
        # Try values from smallest possible
        last = row[-1]
        min_x = last + 1
        max_x = min(max_scope_hint, last + max_scope_hint)  # Upper bound
        
        for x in range(min_x, max_x + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & local_used:
                continue
            if new_diffs & forbidden:
                continue
            
            row.append(x)
            local_used |= new_diffs
            found = True
            break
        
        if not found:
            return None
    
    return row

def _local_search(rows, n_rows, k, deadline, max_iter=5000):
    """
    Local search with targeted row reconstruction and element perturbation.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Build initial used set
    used = set()
    for row in rows:
        used |= _compute_diffs(row)
        
    current_scope = best_scope
    
    for iteration in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        # Decide move type: 70% perturb element, 30% rebuild row
        if random.random() < 0.7:
            # Perturb a single element
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Try to move value - bias towards reducing scope
            if row[-1] == best_scope or row[pos] == best_scope:
                # Try smaller values
                if old_val > 1:
                    new_val = random.randint(1, old_val - 1)
                else:
                    continue
            else:
                # Try random in range, biased towards smaller
                upper = min(current_scope, old_val + 20)
                lower = max(1, old_val - 20)
                new_val = random.randint(lower, upper)
                
            if new_val == old_val:
                continue
                
            # Check order
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
                
            # Compute old diffs involving pos
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, len(row)):
                old_diffs.add(row[j] - old_val)
                
            # Compute new diffs involving pos
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, len(row)):
                new_diffs.add(row[j] - new_val)
                
            # Check conflict
            remaining = used - old_diffs
            if new_diffs & remaining:
                continue
                
            # Valid move
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            
            new_scope = _get_scope(rows)
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
            elif new_scope > current_scope:
                # Revert if scope increased
                row[pos] = old_val
                used = (used - new_diffs) | old_diffs
            else:
                current_scope = new_scope
                
        else:
            # Rebuild a row - target the row with largest max
            max_row_idx = 0
            max_val = -1
            for i, r in enumerate(rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_row_idx = i
            r_idx = max_row_idx
            
            old_row = rows[r_idx]
            old_diffs = _compute_diffs(old_row)
            
            # Remove old row's diffs from global used
            used -= old_diffs
            
            # Try to rebuild with greedy
            new_row = _rebuild_row_greedy(used, k, best_scope)
            
            if new_row is not None:
                new_diffs = _compute_diffs(new_row)
                
                # Check if new diffs conflict with current used (which excludes old row)
                if not (new_diffs & used):
                    rows[r_idx] = new_row
                    used |= new_diffs
                    
                    new_scope = _get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    else:
                        current_scope = new_scope
                else:
                    # Conflict, revert
                    used |= old_diffs
            else:
                # Failed to rebuild, restore old row
                used |= old_diffs
                
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin
    
    # Multi-start strategy with different seeds and skip probabilities
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple initialization strategies with different seeds
    seeds = [42, 123, 456, 789, 1024, 2048, 3072]
    skip_probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for seed, skip_prob in zip(seeds, skip_probs):
        if time.time() > deadline - 30:
            break
        
        rows, _ = _greedy_init(n_rows, k, skip_prob=skip_prob, seed=seed)
        if _is_valid(rows):
            scope = _get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback to pure greedy
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0, seed=42)
        return rows
    
    # Run local search with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        rows = _local_search(best_rows, n_rows, k, deadline, max_iter=8000)
        
        # Final check
        if _is_valid(rows):
            final_scope = _get_scope(rows)
            if final_scope < best_scope:
                best_scope = final_scope
                best_rows = [list(r) for r in rows]
    
    return best_rows
