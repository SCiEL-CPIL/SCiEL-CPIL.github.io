"""Difference triangle set (4,7): minimize the largest entry."""
import time
import random
import copy

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of positive differences within a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate(rows, k):
    """Check that rows form a valid difference triangle set.
    Returns True if valid, False otherwise."""
    total_diffs = 0
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        diffs = _row_diffs(row)
        if len(diffs) != k * (k + 1) // 2:
            return False
        if diffs & seen:
            return False
        seen |= diffs
        total_diffs += len(diffs)
    return total_diffs == len(seen)

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_row(used, k):
    """Build one row greedily: start at 0, pick smallest x whose new diffs don't collide."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _greedy_initial(n_rows, k):
    """Build all rows greedily, returning the initial set."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows

def _all_diffs(rows):
    """Return the full set of differences across all rows."""
    seen = set()
    for row in rows:
        seen |= _row_diffs(row)
    return seen

def _remove_row_diffs(used, row):
    """Remove a row's differences from the used set."""
    diffs = _row_diffs(row)
    used -= diffs
    return used

def _add_row_diffs(used, row):
    """Add a row's differences to the used set."""
    used |= _row_diffs(row)
    return used

def _rebuild_row(row, used, k, max_val):
    """Try to rebuild a row greedily given the current used set (other rows' diffs).
    Returns the new row or None if impossible within max_val."""
    new_used = set(used)
    row_new = [0]
    x = 0
    while len(row_new) < k + 1:
        x += 1
        if x > max_val:
            return None
        new_diffs = {x - a for a in row_new}
        if not new_diffs & new_used:
            row_new.append(x)
            new_used |= new_diffs
    return row_new

def _simulated_annealing(rows, k, time_budget, seed=42):
    """
    Simulated annealing to minimize the scope.
    Moves: pick a row, remove it, rebuild greedily (possibly with perturbation),
    accept if scope decreases or with probability exp(-delta/T).
    """
    rng = random.Random(seed)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    n_rows = len(rows)
    k_plus_1 = k + 1
    max_possible = sum(range(1, k + 1)) * n_rows + n_rows * k  # loose upper bound
    
    # Compute initial used set
    used = _all_diffs(rows)
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    # We'll run until time_budget is exhausted
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Track iterations
    iters = 0
    no_improve = 0
    patience = 5000
    
    while time.time() < deadline - 2.0:
        iters += 1
        
        # Pick a random row to modify
        ri = rng.randrange(n_rows)
        old_row = rows[ri]
        old_scope = _scope(rows)
        
        # Remove this row's diffs from used
        used = _remove_row_diffs(used, old_row)
        
        # Try to rebuild with some randomness
        new_row = None
        attempts = 0
        max_attempts = 20
        while new_row is None and attempts < max_attempts:
            attempts += 1
            # Greedy with perturbation: sometimes skip a valid x to explore
            row_new = [0]
            x = 0
            success = True
            while len(row_new) < k_plus_1:
                x += 1
                if x > old_scope + 10:
                    success = False
                    break
                new_diffs = {x - a for a in row_new}
                if not new_diffs & used:
                    # Accept with some probability to allow exploration
                    if rng.random() < 0.85:
                        row_new.append(x)
                        used |= new_diffs
                    else:
                        # Skip this x, try next
                        pass
                else:
                    pass
            if success and len(row_new) == k_plus_1:
                new_row = row_new
        
        if new_row is None:
            # Restore
            used = _add_row_diffs(used, old_row)
            continue
        
        # Compute new scope
        rows[ri] = new_row
        new_scope = _scope(rows)
        delta = new_scope - old_scope
        
        if delta < 0:
            # Improvement
            rows[ri] = new_row
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Accept with probability exp(-delta/T)
            # Scale temperature based on iteration progress
            elapsed = time.time() - start_time
            progress = min(elapsed / time_budget, 1.0)
            T = T_start * (T_end / T_start) ** progress
            
            if delta == 0:
                # Neutral move, accept with moderate probability
                if rng.random() < 0.3:
                    pass  # keep new_row
                else:
                    rows[ri] = old_row
                    used = _remove_row_diffs(used, new_row)
                    used = _add_row_diffs(used, old_row)
            elif rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                # Accept worse
                no_improve += 1
            else:
                # Reject
                rows[ri] = old_row
                used = _remove_row_diffs(used, new_row)
                used = _add_row_diffs(used, old_row)
        
        # Periodic check for timeout
        if time.time() > deadline - 3.0:
            break
        
        # If stuck for too long, do a larger perturbation
        if no_improve > patience:
            # Try a random restart of one row
            ri2 = rng.randrange(n_rows)
            old_row2 = rows[ri2]
            used = _remove_row_diffs(used, old_row2)
            
            # Fully random rebuild
            row_new = [0]
            x = 0
            success = True
            while len(row_new) < k_plus_1:
                x += 1
                if x > best_scope + 5:
                    success = False
                    break
                new_diffs = {x - a for a in row_new}
                if not new_diffs & used:
                    row_new.append(x)
                    used |= new_diffs
            if success:
                rows[ri2] = row_new
                new_scope2 = _scope(rows)
                if new_scope2 < best_scope:
                    best_scope = new_scope2
                    best_rows = [list(r) for r in rows]
                no_improve = 0
            else:
                used = _add_row_diffs(used, old_row2)
                no_improve = 0  # reset to avoid infinite loop
    
    return best_rows

def _local_search(rows, k, time_budget, seed=123):
    """
    Local search: try swapping individual entries within rows to reduce scope.
    For each row, try removing the largest element and finding a smaller replacement.
    """
    rng = random.Random(seed)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    used = _all_diffs(rows)
    start_time = time.time()
    deadline = start_time + time_budget
    
    n_rows = len(rows)
    k_plus_1 = k + 1
    
    while time.time() < deadline - 1.0:
        improved = False
        for ri in range(n_rows):
            row = rows[ri]
            if len(row) != k_plus_1:
                continue
            
            # Try to reduce the largest entry
            for pos in range(k_plus_1 - 1, 0, -1):
                if row[pos] <= row[pos - 1] + 1:
                    continue
                
                old_val = row[pos]
                # Try values from row[pos-1]+1 to old_val-1
                for new_val in range(row[pos - 1] + 1, old_val):
                    # Check if this new value creates any duplicate diffs
                    # Remove old diffs involving position pos
                    old_diffs = set()
                    for j in range(pos):
                        old_diffs.add(old_val - row[j])
                    
                    # Check new diffs
                    new_diffs = set()
                    for j in range(pos):
                        new_diffs.add(new_val - row[j])
                    
                    # The diffs that are removed
                    removed = old_diffs - new_diffs
                    # The diffs that are added
                    added = new_diffs - old_diffs
                    
                    # Check if added diffs collide with anything else
                    if added & used:
                        continue
                    
                    # This move is valid
                    # Update
                    row[pos] = new_val
                    used -= removed
                    used |= added
                    improved = True
                    
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    
                    break  # Move to next position
                
                if time.time() > deadline - 1.0:
                    return best_rows
        
        if not improved:
            # No improvement found in this pass; try a perturbation
            # Pick a random row and random position, try a random new value
            ri = rng.randrange(n_rows)
            row = rows[ri]
            pos = rng.randrange(1, k_plus_1)
            low = row[pos - 1] + 1
            high = row[pos] + rng.randint(-3, 3)
            if high <= low:
                continue
            
            old_val = row[pos]
            new_val = rng.randint(low, max(low, high))
            if new_val == old_val:
                continue
            
            old_diffs = set()
            for j in range(pos):
                old_diffs.add(old_val - row[j])
            
            new_diffs = set()
            for j in range(pos):
                new_diffs.add(new_val - row[j])
            
            removed = old_diffs - new_diffs
            added = new_diffs - old_diffs
            
            if added & used:
                continue
            
            row[pos] = new_val
            used -= removed
            used |= added
            
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
    
    return best_rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Phase 1: Greedy initial construction
    rows = _greedy_initial(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    if time.time() > deadline - 5.0:
        return best_rows
    
    # Phase 2: Simulated annealing
    sa_time = min(60.0, max(10.0, (deadline - time.time()) * 0.5))
    rows_sa = _simulated_annealing(best_rows, k, sa_time, seed=42)
    scope_sa = _scope(rows_sa)
    if scope_sa < best_scope:
        best_scope = scope_sa
        best_rows = [list(r) for r in rows_sa]
    
    if time.time() > deadline - 10.0:
        return best_rows
    
    # Phase 3: Local search
    ls_time = min(40.0, max(5.0, (deadline - time.time()) * 0.6))
    rows_ls = _local_search(best_rows, k, ls_time, seed=123)
    scope_ls = _scope(rows_ls)
    if scope_ls < best_scope:
        best_scope = scope_ls
        best_rows = [list(r) for r in rows_ls]
    
    if time.time() > deadline - 5.0:
        return best_rows
    
    # Phase 4: Another round of SA from the improved solution
    if time.time() < deadline - 10.0:
        sa_time2 = max(5.0, deadline - time.time() - 5.0)
        rows_sa2 = _simulated_annealing(best_rows, k, sa_time2, seed=999)
        scope_sa2 = _scope(rows_sa2)
        if scope_sa2 < best_scope:
            best_scope = scope_sa2
            best_rows = [list(r) for r in rows_sa2]
    
    # Final validation
    if not _validate(best_rows, k):
        # Fallback to greedy
        best_rows = _greedy_initial(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
