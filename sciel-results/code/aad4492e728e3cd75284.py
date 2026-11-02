import time
import random
import math

_BEST_SCOPE = float('inf')
_BEST_ROWS = None
_DEADLINE = time.time() + 110.0

def _rows_to_diffs(rows):
    """Convert rows to list of difference lists."""
    diffs = []
    for r in rows:
        d = [r[i] - r[i-1] for i in range(1, len(r))]
        diffs.append(d)
    return diffs

def _diffs_to_rows(diffs):
    """Convert difference lists back to rows."""
    rows = []
    for d in diffs:
        r = [0]
        cur = 0
        for x in d:
            cur += x
            r.append(cur)
        rows.append(r)
    return rows

def _check_and_scope(diffs, n_rows, k):
    """
    Check if the set of differences is valid (all global differences distinct) 
    and return the scope (max entry).
    Returns (is_valid, scope).
    """
    used = set()
    max_entry = 0
    for d in diffs:
        cur = 0
        for x in d:
            cur += x
            # Check all differences from current position to previous positions
            # Actually, easier to just compute all pairwise differences for the row
            # But we need to check against global 'used'.
            # Let's reconstruct the row points and check all pairs.
            # This is O(k^2) per row.
            pass
        
    # More efficient check:
    # For each row, generate all pairwise differences.
    # If any difference is already in 'used', invalid.
    # Otherwise, add to 'used'.
    # Scope is max of last element of each row.
    
    global_max = 0
    for d in diffs:
        # Reconstruct row
        row = [0]
        cur = 0
        for x in d:
            cur += x
            row.append(cur)
        
        if cur > global_max:
            global_max = cur
            
        # Check all pairwise differences
        # i < j -> row[j] - row[i]
        row_set = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diff = row[j] - row[i]
                if diff in used:
                    return False, global_max
                if diff in row_set: # Internal collision (should be caught by used if we add immediately, but let's be safe)
                    return False, global_max
                row_set.add(diff)
        
        # Add to global used
        used.update(row_set)
        
    return True, global_max

def _sa_optimize(diffs, n_rows, k, time_limit):
    """
    Simulated Annealing on the difference vectors.
    State: list of difference lists.
    Move: Pick a random row, pick a random position, change the difference by +/- 1 or swap with neighbor?
    Actually, changing a difference changes all subsequent entries in that row.
    This makes the validity check expensive if we do it from scratch.
    
    Alternative Move: 
    1. Pick a row.
    2. Pick a "gap" (difference) in that row.
    3. Increase or decrease it by 1.
    4. This shifts all subsequent points in that row.
    5. We need to update the 'used' set efficiently? 
       No, 'used' is global. If we change one row, we must remove all its old diffs, apply change, and check new diffs against the rest.
       
    Let's implement a specific move:
    - Remove row i's diffs from global 'used'.
    - Modify row i.
    - Check row i's new diffs against remaining 'used'.
    - If valid, update 'used' and accept/reject based on scope.
    """
    
    # Pre-compute current state
    rows = _diffs_to_rows(diffs)
    current_scope = max(max(r) for r in rows)
    
    # Build global used set and per-row sets
    all_used = set()
    row_diffs_sets = []
    for r in rows:
        s = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                s.add(r[j] - r[i])
        row_diffs_sets.append(s)
        all_used.update(s)
        
    best_scope = current_scope
    best_diffs = [list(d) for d in diffs]
    
    start_time = time.time()
    T = 10.0 # Initial temperature
    cooling = 0.9999
    
    while time.time() - start_time < time_limit:
        if time.time() > _DEADLINE:
            break
            
        # Pick a random row
        i = random.randint(0, n_rows - 1)
        d = diffs[i]
        
        # Pick a random index in the difference list to modify
        # d has length k
        if k == 0: continue
        j = random.randint(0, k - 1)
        
        # Choose delta: -1, +1, or sometimes larger jumps
        # Small jumps are better for fine-tuning scope
        if random.random() < 0.8:
            delta = random.choice([-1, 1])
        else:
            delta = random.choice([-2, -3, 2, 3])
            
        new_val = d[j] + delta
        if new_val <= 0:
            continue
            
        # Save old state of row i
        old_d = list(d)
        d[j] = new_val
        
        # Reconstruct row i
        row = [0]
        cur = 0
        for x in d:
            cur += x
            row.append(cur)
            
        new_scope = cur
        
        # Check validity of new row against global used (excluding row i)
        # We need to check if any new diff is in (all_used - row_diffs_sets[i])
        # Actually, easier:
        # 1. Remove old row diffs from all_used
        # 2. Check new row diffs against all_used
        # 3. If valid, add new row diffs to all_used
        # 4. If invalid, revert all_used to include old row diffs (and revert d)
        
        old_set = row_diffs_sets[i]
        for x in old_set:
            all_used.discard(x)
            
        valid = True
        new_set = set()
        for a in range(len(row)):
            for b in range(a+1, len(row)):
                diff = row[b] - row[a]
                if diff in all_used:
                    valid = False
                    break
                new_set.add(diff)
            if not valid:
                break
                
        if valid:
            # Accept if scope improves, or with probability exp(-delta_scope / T)
            if new_scope < current_scope:
                # Always accept improvement
                # Update best
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_diffs = [list(dd) for dd in diffs]
                    # Update global state
                    all_used.update(new_set)
                    row_diffs_sets[i] = new_set
                    current_scope = new_scope
                else:
                    # Accept with probability? 
                    # If scope didn't improve but we are in SA, we might accept worse to escape local minima.
                    # But scope is the objective. 
                    # Let's only accept if new_scope <= current_scope for now to keep it simple?
                    # No, SA needs to accept worse.
                    # But "worse" means larger scope.
                    delta_scope = new_scope - current_scope
                    if delta_scope < 0:
                        accept = True
                    else:
                        accept = random.random() < math.exp(-delta_scope / T)
                        
                    if accept:
                        all_used.update(new_set)
                        row_diffs_sets[i] = new_set
                        current_scope = new_scope
                    else:
                        # Reject: revert all_used
                        all_used.update(old_set)
                        d[j] = old_d[j]
            else:
                # Scope worsened
                delta_scope = new_scope - current_scope
                accept = random.random() < math.exp(-delta_scope / T)
                if accept:
                    all_used.update(new_set)
                    row_diffs_sets[i] = new_set
                    current_scope = new_scope
                else:
                    all_used.update(old_set)
                    d[j] = old_d[j]
        else:
            # Invalid: revert
            all_used.update(old_set)
            d[j] = old_d[j]
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_scope, best_diffs

def construct(n_rows, k):
    global _BEST_SCOPE, _BEST_ROWS, _DEADLINE
    
    _BEST_SCOPE = float('inf')
    _BEST_ROWS = None
    _DEADLINE = time.time() + 110.0
    
    rng = random.Random()
    
    # 1. Initial greedy construction (fast)
    # Use a bitmask greedy to get a valid starting point
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
        rows.append(row)
        
    initial_diffs = _rows_to_diffs(rows)
    initial_scope = max(max(r) for r in rows)
    
    _BEST_SCOPE = initial_scope
    _BEST_ROWS = [list(r) for r in rows]
    
    # 2. Run Simulated Annealing
    # Allocate time
    start_sa = time.time()
    time_budget = 100.0 # Leave 10s for safety
    
    best_scope, best_diffs = _sa_optimize(initial_diffs, n_rows, k, time_budget)
    
    if best_scope < _BEST_SCOPE:
        _BEST_SCOPE = best_scope
        _BEST_ROWS = _diffs_to_rows(best_diffs)
        
    # 3. Try a few more restarts if time permits
    # The SA above is one run. We can run it again with different initial conditions or just let it run longer?
    # The _sa_optimize function has an internal loop. 
    # If we have time left, we can run another SA from the best found so far.
    
    remaining = _DEADLINE - time.time()
    if remaining > 5.0 and _BEST_ROWS is not None:
        # Restart from best
        start_diffs = _rows_to_diffs(_BEST_ROWS)
        best_scope2, best_diffs2 = _sa_optimize(start_diffs, n_rows, k, remaining - 5.0)
        if best_scope2 < _BEST_SCOPE:
            _BEST_SCOPE = best_scope2
            _BEST_ROWS = _diffs_to_rows(best_diffs2)
            
    if _BEST_ROWS is not None:
        return _BEST_ROWS
    return rows

# Fallback if something goes wrong
# The construct function should always return valid rows.
# The greedy part ensures validity.
# The SA part maintains validity by checking before accepting.
