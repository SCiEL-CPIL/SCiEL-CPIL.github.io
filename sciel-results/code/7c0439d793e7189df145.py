"""Constraint-satisfaction search for the difference-triangle problem using Simulated Annealing."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_fallback(n_rows, k):
    """Original greedy: always valid, used as safety net."""
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

def _scope(rows):
    return max(max(r) for r in rows)

def _compute_row_diffs(row):
    """Compute set of positive differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s kill
    rng = random.Random(42)
    
    # Phase 1: Greedy Initialization
    best_rows = _greedy_fallback(n_rows, k)
    best_scope = _scope(best_rows)
    
    # State for Simulated Annealing
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    # Pre-compute global used differences for fast validity checks
    # We need to be careful: the set of used diffs must be consistent with current_rows.
    # We will maintain a global set `global_used` and update it incrementally.
    
    def get_global_used(rows):
        used = set()
        for r in rows:
            used |= _compute_row_diffs(r)
        return used

    global_used = get_global_used(current_rows)
    
    # SA Parameters
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.9995
    steps_per_iter = 100 # Try to do many moves per "temperature" step if time permits, 
                         # but actually we just loop until deadline.
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Perform a move
        # Choose a random row
        r_idx = rng.randrange(n_rows)
        row = current_rows[r_idx]
        
        # Choose a random index to modify (0 is fixed)
        # We will try to shift row[i] by delta
        i = rng.randrange(1, len(row))
        
        # Determine delta
        # Bias towards reducing scope if possible, but allow exploration
        # If row[i] is the max, try to reduce it more
        if row[i] == current_scope:
            delta = rng.choice([-1, -2, -3, -5, -10, 1, 2])
        else:
            delta = rng.choice([-1, 1, -2, 2, -3, 3, -5, 5])
        
        old_val = row[i]
        new_val = old_val + delta
        
        # Check bounds and increasing order
        if new_val <= row[i-1]:
            continue
        if i < len(row) - 1 and new_val >= row[i+1]:
            continue
        if new_val < 0:
            continue
            
        # Check validity of differences
        # 1. Remove old differences involving index i
        old_diffs = set()
        for j in range(len(row)):
            if j == i: continue
            d = abs(row[j] - row[i])
            old_diffs.add(d)
        
        # 2. Check if new differences collide with global_used (excluding old_diffs)
        #    or with each other / within the row
        new_diffs = set()
        valid = True
        for j in range(len(row)):
            if j == i: continue
            d = abs(row[j] - new_val)
            # If this difference was already in the row (from another pair) or globally,
            # we need to ensure we are not creating a duplicate *after* removing the old ones.
            # Actually, simpler: 
            # The set of all diffs in the row will be: (old_row_diffs - old_diffs) U new_diffs
            # We need (old_row_diffs - old_diffs) U new_diffs to be disjoint from 
            # (global_used - old_diffs)
            
            # Let's just compute the new row diffs and check against global.
            # For efficiency, we can check collision on the fly.
            # Collision 1: new_diff d is in new_diffs (duplicate within new set)
            if d in new_diffs:
                valid = False
                break
            # Collision 2: new_diff d is in global_used AND d is NOT in old_diffs
            # (If d is in old_diffs, it's being removed, so it's free)
            if d in global_used and d not in old_diffs:
                valid = False
                break
            new_diffs.add(d)
            
        if not valid:
            continue
            
        # Move is valid!
        # Calculate change in scope
        old_scope = current_scope
        # Update row
        row[i] = new_val
        new_scope = max(row)
        # The global scope is max of all rows. 
        # If this row was the only one with old_scope, and new_scope < old_scope, global scope might drop.
        # If new_scope > old_scope, global scope might rise.
        # To be safe and fast, we compute global scope only if this row's max changed significantly or is the max.
        # Actually, just compute max of all rows is O(n_rows). n_rows is small (7).
        new_global_scope = 0
        for r in current_rows:
            m = max(r)
            if m > new_global_scope:
                new_global_scope = m
        
        delta_E = new_global_scope - old_scope
        
        # Acceptance criterion
        accept = False
        if delta_E <= 0:
            accept = True
        else:
            # Boltzmann factor
            # Avoid division by zero or overflow
            if T > 1e-9:
                prob = math.exp(-delta_E / T)
                accept = rng.random() < prob
            else:
                accept = False
                
        if accept:
            current_scope = new_global_scope
            # Update global_used
            global_used -= old_diffs
            global_used |= new_diffs
            
            # Update best
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
                
            # Cool down
            T = max(T_min, T * cooling_rate)
        else:
            # Reject: Revert row
            row[i] = old_val
            # No need to update global_used as it wasn't changed
            # Cool down slightly even on rejection? Standard SA cools every step.
            T = max(T_min, T * cooling_rate)
            
        # Safety check: ensure we don't run out of time in the middle of a move
        if time.time() > deadline - 1.0:
            break
            
    return best_rows

# EVOLVE-BLOCK-END
