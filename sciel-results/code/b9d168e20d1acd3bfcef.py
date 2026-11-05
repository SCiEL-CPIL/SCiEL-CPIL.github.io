import time
import random
import math
import sys

def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                all_diffs.append(row[j] - ri)
    return len(all_diffs) == len(set(all_diffs))

def _greedy_construct(n_rows, num_elements):
    """
    Fast greedy construction to get a valid initial solution.
    Returns rows and the initial scope.
    """
    rows = []
    used_mask = 0
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_mask) and len(new_diffs) == len(set(new_diffs)):
                row.append(x)
                for d in new_diffs:
                    used_mask |= (1 << d)
        rows.append(row)
    return rows, _scope(rows)

def _simulated_annealing(rows, num_elements, n_rows, start_time, deadline):
    """
    Simulated Annealing to minimize the scope.
    State: list of rows.
    Moves:
    1. Move a single element in a row (up or down, maintaining order).
    2. Swap two rows (if they have different structures, though usually not helpful for scope unless one is much smaller, but here all rows are same size).
    3. Shift a row: add/subtract a constant to all elements except 0? No, 0 is fixed.
       Actually, shifting a row up increases scope, shifting down might violate increasing order if 0 is fixed.
       So "move element" is the primary local move.
    4. "Swap" elements between two rows at the same position? 
       Let's stick to:
       - Move element (i, j) to new value v.
       - Swap two entire rows (useless if symmetric, but maybe helps if we consider row ordering? No, rows are independent).
       - Let's try: Swap two elements at the same position in two different rows.
         If row A has a_j and row B has b_j. Swap them.
         Must maintain increasing order within rows.
    """
    
    # Precompute masks for current state
    row_masks = [_row_mask(r) for r in rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
        
    current_scope = _scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # Temperature schedule
    # Start high enough to allow scope increases
    T = 100.0
    T_min = 1e-6
    cooling = 0.995
    
    rng = random.Random(42)
    
    steps = 0
    max_steps = 100000 # Safety cap
    
    while time.time() < deadline - 5 and T > T_min and steps < max_steps:
        steps += 1
        
        # Choose move type
        move_type = rng.random()
        
        new_rows = [r[:] for r in rows]
        valid_move = True
        delta_scope = 0
        affected_rows = []
        
        if move_type < 0.8:
            # Move a single element
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, num_elements - 1)
            
            row = new_rows[r_idx]
            old_val = row[pos]
            
            # Determine range for new value
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < num_elements - 1 else current_scope + 10 # Allow some expansion
            
            if low > high:
                continue
            
            # Sample new value
            # Bias towards lower values to reduce scope
            # But also allow higher values to escape local minima
            if rng.random() < 0.7:
                # Try to lower
                if old_val > low:
                    new_val = rng.randint(low, old_val - 1)
                else:
                    new_val = old_val
            else:
                # Try to increase or stay
                new_val = rng.randint(old_val, high)
            
            if new_val == old_val:
                continue
            
            # Check validity of the move
            # Remove old diffs for this row, add new diffs
            old_row_mask = row_masks[r_idx]
            
            # Calculate new diffs for the row
            # We can do this efficiently by just checking the new value against existing ones
            # But we need the full mask for the row to update global_mask correctly.
            # Since k is small (6 elements -> 7 items -> 21 diffs), recalculating the row mask is fast.
            
            row[pos] = new_val
            
            # Check increasing order
            if pos > 0 and row[pos] <= row[pos-1]:
                valid_move = False
            elif pos < num_elements - 1 and row[pos] >= row[pos+1]:
                valid_move = False
            else:
                # Calculate new row mask
                new_row_mask = _row_mask(row)
                
                # Check for collisions with other rows
                # global_mask_without_this_row = global_mask & ~old_row_mask
                other_mask = global_mask & ~old_row_mask
                
                if new_row_mask & other_mask:
                    valid_move = False
                else:
                    # Move is valid
                    # Update state
                    row_masks[r_idx] = new_row_mask
                    global_mask = (global_mask & ~old_row_mask) | new_row_mask
                    affected_rows = [r_idx]
                    
                    # Calculate delta scope
                    old_scope_val = row[-1]
                    new_scope_val = new_rows[r_idx][-1]
                    # Note: row[-1] might not be the element we moved, but scope depends on max element.
                    # We need to recompute scope or track max.
                    # For efficiency, let's just compute new scope if it might have changed.
                    # Actually, scope is max of all row[-1].
                    # Let's store last elements.
                    pass
                    
        elif move_type < 0.95:
            # Swap two elements at the same position in two different rows
            r1, r2 = rng.sample(range(n_rows), 2)
            pos = rng.randint(1, num_elements - 1)
            
            v1 = rows[r1][pos]
            v2 = rows[r2][pos]
            
            if v1 == v2:
                continue
            
            # Try swapping
            new_rows[r1][pos] = v2
            new_rows[r2][pos] = v1
            
            # Check order in r1
            if pos > 0 and new_rows[r1][pos] <= new_rows[r1][pos-1]:
                valid_move = False
            elif pos < num_elements - 1 and new_rows[r1][pos] >= new_rows[r1][pos+1]:
                valid_move = False
            # Check order in r2
            elif pos > 0 and new_rows[r2][pos] <= new_rows[r2][pos-1]:
                valid_move = False
            elif pos < num_elements - 1 and new_rows[r2][pos] >= new_rows[r2][pos+1]:
                valid_move = False
            else:
                # Check collisions
                m1_new = _row_mask(new_rows[r1])
                m2_new = _row_mask(new_rows[r2])
                
                # Other rows mask
                other_mask = global_mask & ~row_masks[r1] & ~row_masks[r2]
                
                if (m1_new & m2_new) or (m1_new & other_mask) or (m2_new & other_mask):
                    valid_move = False
                else:
                    # Valid
                    row_masks[r1] = m1_new
                    row_masks[r2] = m2_new
                    global_mask = (global_mask & ~row_masks[r1]) # Wait, row_masks[r1] was old
                    # Let's redo global mask update carefully
                    old_m1 = _row_mask(rows[r1]) # Recalculating is safer if we didn't store old
                    old_m2 = _row_mask(rows[r2])
                    # Actually we have row_masks from before the swap attempt in this branch?
                    # No, we modified new_rows. row_masks is still old.
                    # So:
                    global_mask = (global_mask & ~old_m1 & ~old_m2) | m1_new | m2_new
                    affected_rows = [r1, r2]
        
        else:
            # "Kick" move: Randomly perturb a row slightly to escape deep local minima
            # Pick a row and randomize one element significantly
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, num_elements - 1)
            
            row = new_rows[r_idx]
            old_val = row[pos]
            
            low = row[pos - 1] + 1
            high = current_scope + 20 # Allow expansion
            
            if low > high:
                continue
                
            new_val = rng.randint(low, high)
            
            if new_val == old_val:
                continue
            
            row[pos] = new_val
            
            if pos > 0 and row[pos] <= row[pos-1]:
                valid_move = False
            elif pos < num_elements - 1 and row[pos] >= row[pos+1]:
                valid_move = False
            else:
                new_row_mask = _row_mask(row)
                other_mask = global_mask & ~row_masks[r_idx]
                
                if new_row_mask & other_mask:
                    valid_move = False
                else:
                    row_masks[r_idx] = new_row_mask
                    global_mask = (global_mask & ~_row_mask(rows[r_idx])) | new_row_mask
                    affected_rows = [r_idx]
                    
        if not valid_move:
            # Revert changes if we modified new_rows but found it invalid
            # Since we modify new_rows in place, we need to revert.
            # To avoid complexity, let's only accept if valid.
            # But we need to calculate delta_scope for acceptance.
            # If invalid, we just continue.
            continue
            
        # Calculate new scope
        new_scope = max(r[-1] for r in new_rows)
        delta_scope = new_scope - current_scope
        
        # Acceptance
        if delta_scope <= 0:
            # Always accept improvements or neutral moves
            accept = True
        else:
            # Accept with probability exp(-delta/T)
            # Use log for stability? T is small later.
            # If T is very small, exp(-large) is 0.
            try:
                prob = math.exp(-delta_scope / T)
            except OverflowError:
                prob = 0.0
            accept = rng.random() < prob
            
        if accept:
            rows = new_rows
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        
        # Cool down
        T *= cooling
        
    return best_rows, best_scope

def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 135.0
    
    num_elements = k + 1
    
    # Phase 1: Greedy initialization
    initial_rows, initial_scope = _greedy_construct(n_rows, num_elements)
    
    best_rows = initial_rows
    best_scope = initial_scope
    
    # Phase 2: Simulated Annealing
    # Run SA with a time limit
    sa_deadline = DEADLINE - 10
    sa_rows, sa_scope = _simulated_annealing(initial_rows, num_elements, n_rows, start_time, sa_deadline)
    
    if sa_scope < best_scope:
        best_scope = sa_scope
        best_rows = sa_rows
    
    # Phase 3: Local Search Refinement (Greedy descent)
    # Try to push elements down greedily
    current_rows = [r[:] for r in best_rows]
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
    
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        for r_idx in range(n_rows):
            row = current_rows[r_idx]
            for pos in range(num_elements - 1, 0, -1):
                old_val = row[pos]
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < num_elements - 1 else old_val
                
                # Try to find a smaller valid value
                # Check from low to old_val - 1
                for new_val in range(low, old_val):
                    row[pos] = new_val
                    new_row_mask = _row_mask(row)
                    other_mask = global_mask & ~row_masks[r_idx]
                    
                    if not (new_row_mask & other_mask):
                        # Accept
                        row_masks[r_idx] = new_row_mask
                        global_mask = (global_mask & ~_row_mask(current_rows[r_idx])) # Wait, row_masks[r_idx] is new
                        # Correct update:
                        old_m = _row_mask([row[i] if i != pos else old_val for i in range(num_elements)])
                        global_mask = (global_mask & ~old_m) | new_row_mask
                        row_masks[r_idx] = new_row_mask
                        improved = True
                        break
                    else:
                        row[pos] = old_val
                        # If we tried the lowest available and it failed, higher ones might fail too?
                        # Not necessarily, but usually lower is better. 
                        # However, a slightly higher value might open up space for other elements?
                        # No, we are fixing other elements. So if low fails, we stop?
                        # Actually, if low fails due to collision, a higher value might not collide.
                        # But we want to minimize, so we try all.
                        # Optimization: if low fails, we can break if we assume monotonicity? No.
                        # Let's just continue checking, but it's slow.
                        # Given small k, it's fine.
                
                if time.time() > DEADLINE - 5:
                    break
            if time.time() > DEADLINE - 5:
                break
    
    final_scope = _scope(current_rows)
    if final_scope < best_scope:
        best_scope = final_scope
        best_rows = current_rows
    
    # Final validation
    if not _is_valid(best_rows):
        # Fallback to greedy if something went wrong
        best_rows, best_scope = _greedy_construct(n_rows, num_elements)
        
    return best_rows

def construct(n_rows, k):
    return _construct(n_rows, k)
