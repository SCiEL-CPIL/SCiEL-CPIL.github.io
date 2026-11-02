import time
import random
import sys

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

def _greedy_init(n_rows, k):
    """
    Greedy initialization: build rows one by one.
    For each row, pick the smallest possible next integer that doesn't create a conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Check if adding x creates a conflict with existing elements in this row
            # and with the global used set
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _rebuild_row(row_idx, rows, used_diffs, k, max_scope_hint):
    """
    Try to rebuild row[row_idx] to minimize its maximum element,
    given the set of used differences from other rows.
    Returns the new row if successful, else None.
    """
    # The "forbidden" differences are those used by other rows.
    # We want to find a sequence 0 = a_0 < a_1 < ... < a_k such that
    # all diffs a_j - a_i are NOT in forbidden_diffs.
    
    forbidden = set(used_diffs)
    
    # We use a DFS/backtracking to find the smallest possible last element.
    # Since k is small (5), the depth is 6.
    # We can bound the search by max_scope_hint.
    
    best_row = None
    best_last = max_scope_hint + 10 # Start with a high bound
    
    # To make it faster, we can use a greedy approach with limited backtracking
    # or just a simple greedy that tries to keep numbers small.
    # Given the time constraint, a pure greedy rebuild is faster and often sufficient.
    # Let's try a greedy rebuild first.
    
    row = [0]
    current_max = 0
    
    # We try to fill the row greedily.
    # To encourage smaller numbers, we iterate x from 1 upwards.
    # However, we might get stuck. 
    # A better heuristic: try to pick the smallest x that doesn't conflict.
    
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > best_last:
            break
        new_diffs = {x - a for a in row}
        if not new_diffs & forbidden:
            row.append(x)
            # We don't update forbidden here because we are just checking validity
            # against the OTHER rows. The internal diffs of this row must be distinct 
            # from each other too, but since x is increasing and we check against 
            # existing row elements, and the set 'forbidden' doesn't contain 
            # this row's internal diffs yet, we must ensure internal uniqueness.
            # Wait, the condition is "all differences ... are distinct over all rows".
            # This includes differences within the same row.
            # The greedy check `new_diffs & forbidden` only checks against OTHER rows.
            # We must also ensure `new_diffs` are distinct from each other?
            # No, `new_diffs` are differences between x and previous elements.
            # Are they distinct from each other? 
            # If row is [0, 2, 5], diffs for 5 are {5, 3}. Distinct.
            # If row is [0, 3, 5], diffs for 5 are {5, 2}. Distinct.
            # In a strictly increasing sequence, differences to a fixed new element x 
            # are x - a_i. Since a_i are distinct, x - a_i are distinct.
            # So internal uniqueness is guaranteed by strict increase.
            
            # However, we must ensure that the NEW diffs don't conflict with 
            # ALREADY chosen diffs in THIS row.
            # My current `forbidden` set does NOT include the diffs of the current row being built.
            # So I need to maintain a local set of used diffs for this row.
            
            pass # Logic flaw in simple greedy above if I don't track local diffs
            
    # Let's restart the rebuild logic correctly.
    
    row = [0]
    local_used = set()
    x = 0
    success = False
    
    # We want to minimize the last element.
    # Greedy: pick smallest valid x.
    # This might not yield the global minimum for the row, but it's fast.
    
    while len(row) < k + 1:
        x += 1
        if x > best_last:
            break
        new_diffs = {x - a for a in row}
        # Check against local used diffs
        if new_diffs & local_used:
            continue
        # Check against global forbidden diffs
        if new_diffs & forbidden:
            continue
        
        row.append(x)
        local_used |= new_diffs
        
    if len(row) == k + 1:
        if row[-1] < best_last:
            best_last = row[-1]
            best_row = list(row)
            
    return best_row

def _sa_optimize(rows, n_rows, k, deadline):
    """
    Optimized Simulated Annealing with Row Rebuilds.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Build initial used set
    used = set()
    for row in rows:
        used |= _compute_diffs(row)
        
    current_scope = best_scope
    T = 100.0
    cooling = 0.998
    min_T = 0.5
    
    # Precompute diff sets for each row for faster updates?
    # No, just recompute on move.
    
    while T > min_T and time.time() < deadline:
        # Decide move type: 70% perturb element, 30% rebuild row
        if random.random() < 0.7:
            # Perturb a single element
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Try to move value closer to 0 or within a range
            # Bias towards reducing scope
            if random.random() < 0.5:
                # Try smaller
                new_val = random.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                # Try random in range
                new_val = random.randint(1, current_scope + 5)
                
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
            # The set of all diffs is `used`.
            # The diffs that remain from this row are `used - old_diffs`.
            # Note: `used` contains diffs from ALL rows.
            # `old_diffs` are specific to this row.
            # If we remove `old_diffs` from `used`, we get diffs from other rows + other diffs in this row.
            # We must ensure `new_diffs` do not intersect with this remaining set.
            
            remaining = used - old_diffs
            if new_diffs & remaining:
                T *= cooling
                continue
                
            # Valid move
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            
            new_scope = _get_scope(rows)
            delta = new_scope - current_scope
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
            else:
                # Revert
                row[pos] = old_val
                used = (used - new_diffs) | old_diffs
                
        else:
            # Rebuild a row
            r_idx = random.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = _compute_diffs(old_row)
            
            # Remove old row's diffs from global used
            used -= old_diffs
            
            # Try to rebuild
            new_row = _rebuild_row(r_idx, rows, used, k, best_scope)
            
            if new_row is not None:
                new_diffs = _compute_diffs(new_row)
                
                # Check if new diffs conflict with current used (which excludes old row)
                if not (new_diffs & used):
                    # Accept if scope is better or equal, or with probability
                    # Note: _rebuild_row minimizes the last element of the row,
                    # but the global scope is max of all rows.
                    # Rebuilding one row might reduce the global scope if that row was the bottleneck.
                    
                    rows[r_idx] = new_row
                    used |= new_diffs
                    
                    new_scope = _get_scope(rows)
                    delta = new_scope - current_scope
                    
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in rows]
                    else:
                        # Revert
                        rows[r_idx] = old_row
                        used = (used - new_diffs) | old_diffs
                else:
                    # Conflict, revert
                    used |= old_diffs
            else:
                # Failed to rebuild, restore old row
                used |= old_diffs
                
        T *= cooling
        
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 145.0 # Safety margin
    
    # Initialize
    rows = _greedy_init(n_rows, k)
    
    # Verify
    if not _is_valid(rows):
        return rows
        
    # Optimize
    rows = _sa_optimize(rows, n_rows, k, deadline)
    
    # Final check
    if not _is_valid(rows):
        return _greedy_init(n_rows, k)
        
    return rows
