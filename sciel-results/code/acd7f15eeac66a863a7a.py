"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Helper to check if a move is valid and compute the new max
    # We maintain a global set of used differences for O(1) conflict checks.
    # When we change a row, we must remove its old differences and add new ones.
    
    def greedy_construct():
        """Generate a valid starting solution using pure greedy."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x creates a conflict
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows, used

    # Initialize with a greedy solution
    rows, used_diffs = greedy_construct()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max
    
    # Simulated Annealing Parameters
    # We want to minimize the maximum entry.
    # Temperature starts high and decays.
    initial_temp = 50.0
    final_temp = 0.1
    cooling_rate = 0.995
    
    # Precompute some info
    num_elements = k + 1
    total_elements = n_rows * num_elements
    
    # Helper to get the differences for a specific row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to check if a row is strictly increasing and starts at 0
    # (We assume rows[0] is always 0 and we only move elements > 0, so this is usually true
    # unless we move an element to 0 or below, which we prevent)
    
    # Main SA Loop
    temp = initial_temp
    rng = random.Random(42)
    
    # We track the current state
    current_rows = [r[:] for r in rows]
    current_used = set(used_diffs)
    
    # To speed up conflict checking, we can't just use a global set for "what is allowed"
    # because we need to know if a *specific* difference is used by *other* rows.
    # Actually, the constraint is: all differences across ALL rows must be distinct.
    # So `current_used` contains ALL differences from ALL rows.
    # If I change row `r_idx`, I must:
    # 1. Remove all differences of the old row `r_idx` from `current_used`.
    # 2. Propose a new value for an element in `r_idx`.
    # 3. Check if the new differences (within the new row) are in `current_used` (which now excludes old row diffs).
    # 4. If valid, add new differences to `current_used`.
    
    # Optimization: Storing the set of diffs for each row separately might be faster for removal?
    # No, removing from a global set is O(1) per element.
    
    steps = 0
    while time.time() < deadline:
        # Select a random row
        r_idx = rng.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # We cannot change row[0] (must be 0)
        if len(row) <= 1:
            continue
            
        # Select a random index to change (1 to k)
        e_idx = rng.randint(1, len(row) - 1)
        old_val = row[e_idx]
        
        # Propose a new value
        # Strategy: Move slightly towards 0 or away?
        # To reduce max, we generally want to move elements down.
        # But we might need to move up to allow others down later.
        # Let's try a random delta.
        # If we are near the max, bias towards decreasing.
        
        # Determine a range for the new value
        # It must be > row[e_idx-1] and < row[e_idx+1] (if exists)
        lower_bound = row[e_idx - 1] + 1
        upper_bound = row[e_idx + 1] - 1 if e_idx < len(row) - 1 else old_val + 10 # Arbitrary upper
        
        # If we want to reduce scope, we should focus on the last element or elements that contribute to the max.
        # But moving any element changes diffs.
        
        # Let's try a random integer in a window around old_val
        # Window size depends on temperature?
        # Or just try old_val - 1, old_val + 1, etc.
        
        # A better move: try to shrink the row.
        # If e_idx is the last element, try decreasing it.
        # If e_idx is not last, decreasing it might violate upper bound.
        
        # Let's generate a candidate
        # With prob 0.5, try to decrease (if possible)
        # With prob 0.5, try to increase (if possible)
        # Or random offset.
        
        # Simple approach: random offset in [-5, 5] scaled by temp?
        # No, let's be smarter.
        
        # If e_idx == len(row) - 1 (last element):
        #   Try to decrease it.
        # Else:
        #   Try random shift.
        
        if e_idx == len(row) - 1:
            # Try to reduce the max
            # Try decreasing by 1, 2, 3... up to some limit
            # Or pick a random candidate between lower_bound and old_val - 1
            if old_val > lower_bound:
                # Pick a random value in [lower_bound, old_val - 1]
                # But bias towards smaller?
                # Let's just pick a random one.
                new_val = rng.randint(lower_bound, old_val - 1)
            else:
                # Can't decrease, try increase? (Unlikely to help max, but maybe for SA dynamics)
                if old_val < upper_bound:
                    new_val = rng.randint(old_val + 1, upper_bound)
                else:
                    continue
        else:
            # Middle element
            # Try random shift
            # Range: [lower_bound, upper_bound]
            # Pick a random value
            if lower_bound >= upper_bound:
                continue
            # Bias towards keeping it close?
            # Or just uniform random in the valid range?
            # Uniform random in [lower_bound, upper_bound] might be too wide.
            # Let's try a small random walk.
            delta = rng.randint(-3, 3)
            new_val = old_val + delta
            if new_val < lower_bound or new_val > upper_bound:
                # Clamp or reject
                if new_val < lower_bound:
                    new_val = lower_bound
                elif new_val > upper_bound:
                    new_val = upper_bound
                if new_val == old_val:
                    continue
        
        # Check validity
        # 1. Remove old row diffs from global set
        old_diffs = get_row_diffs(row)
        for d in old_diffs:
            current_used.discard(d)
        
        # 2. Apply change
        row[e_idx] = new_val
        
        # 3. Check new row diffs against global set
        new_diffs = get_row_diffs(row)
        valid = True
        for d in new_diffs:
            if d in current_used:
                valid = False
                break
        
        if valid:
            # Accept the move
            # Add new diffs to global set
            current_used.update(new_diffs)
            
            new_max = max(max(r) for r in current_rows)
            
            # Update best
            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in current_rows]
            
            # SA Acceptance
            # We always accept improvements.
            # For worsening moves, accept with probability exp(-delta_E / T)
            # Here "Energy" is the max entry.
            # If new_max > current_max, delta_E = new_max - current_max
            # If new_max <= current_max, accept.
            
            if new_max <= current_max:
                current_max = new_max
            else:
                delta_e = new_max - current_max
                # Probability of accepting worse move
                # p = exp(-delta_e / temp)
                if temp > 0:
                    p = math.exp(-delta_e / temp)
                    if rng.random() < p:
                        current_max = new_max
                    else:
                        # Reject: revert
                        row[e_idx] = old_val
                        # Re-add old diffs, remove new diffs (which are not in set yet)
                        # Wait, if we reject, we must restore the state.
                        # We removed old_diffs from current_used.
                        # We did NOT add new_diffs to current_used (only checked).
                        # So we just need to re-add old_diffs.
                        current_used.update(old_diffs)
                else:
                    # Reject
                    row[e_idx] = old_val
                    current_used.update(old_diffs)
        else:
            # Invalid move, revert
            row[e_idx] = old_val
            current_used.update(old_diffs)
            
        # Cool down
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp
            
        steps += 1
        # Break if time is up
        if time.time() > deadline - 5.0:
            break

    return best_rows
# EVOLVE-BLOCK-END
