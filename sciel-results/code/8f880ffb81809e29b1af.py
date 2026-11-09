"""Optimized solver for Difference Triangle Set (n_rows, k) using Simulated Annealing."""
import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0
    
    k_plus_1 = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        used_mask = 0
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if (used_mask >> d) & 1:
                        return False
                    used_mask |= (1 << d)
        return True

    def row_diff_mask(row):
        mask = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def try_build_row(used_mask, max_val, k_plus_1, rng):
        row = [0]
        for _ in range(k_plus_1 - 1):
            found = False
            start_x = row[-1] + 1
            for x in range(start_x, max_val + 1):
                conflict = False
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    if d > max_val:
                        conflict = True
                        break
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    new_diffs_mask |= (1 << d)
                
                if not conflict:
                    row.append(x)
                    used_mask |= new_diffs_mask
                    found = True
                    break
            
            if not found:
                return None
        
        return row

    def solve_for_scope(max_scope, rng):
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        used_mask = 0
        rows = [None] * n_rows
        
        for r_idx in row_order:
            row = try_build_row(used_mask, max_scope, k_plus_1, rng)
            if row is None:
                return None
            rows[r_idx] = row
            used_mask |= row_diff_mask(row)
            
        return rows

    # Phase 1: Get a baseline solution using the existing greedy method
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))
    initial_scope = max(lb + 50, 150)
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    for attempt in range(100):
        if time.time() > time_limit - 60: # Reserve time for SA
            break
        res = solve_for_scope(initial_scope, rng)
        if res is not None:
            best_rows = res
            best_scope = get_scope(res)
            break
    
    if best_rows is None:
        # Fallback
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k_plus_1:
                x = row[-1] + 1
                while True:
                    conflict = False
                    for a in row:
                        d = x - a
                        if (used_mask >> d) & 1:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used_mask |= (1 << (x - a))
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        best_scope = get_scope(best_rows)
        return best_rows

    # Phase 2: Simulated Annealing to minimize scope
    # State: best_rows
    # We work on a copy
    current_rows = [row[:] for row in best_rows]
    current_scope = best_scope
    
    # Precompute row diff masks for fast updates
    row_masks = [row_diff_mask(row) for row in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
        
    def get_new_row_diffs(row, idx, new_val):
        """Calculate the diff mask for a row if element at idx is changed to new_val."""
        n = len(row)
        # We need to check if the new row is valid internally (distinct diffs)
        # and return the mask.
        # To be efficient, we can just compute the mask and check for bit count?
        # No, we need to ensure no duplicates.
        # A mask doesn't tell us if there were duplicates (bit count < K).
        
        # Construct new row
        new_row = row[:]
        new_row[idx] = new_val
        
        # Check strictly increasing
        for i in range(n-1):
            if new_row[i] >= new_row[i+1]:
                return None, None
        
        # Compute diffs
        mask = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                d = new_row[j] - new_row[i]
                if d > 1000: # Safety
                    return None, None
                if (mask >> d) & 1:
                    return None, None # Duplicate diff in this row
                mask |= (1 << d)
                count += 1
        
        if count != total_diffs_per_row:
            return None, None
            
        return new_row, mask

    # SA Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    steps = 0
    max_steps = 200000
    
    rng_sa = random.Random(123)
    
    while T > T_min and steps < max_steps:
        if time.time() > time_limit:
            break
            
        # Pick a row and an index to change
        r = rng_sa.randint(0, n_rows - 1)
        # Bias towards changing the last element to reduce scope?
        # Or random.
        # Let's try random index 1..k (index 0 is always 0)
        j = rng_sa.randint(1, k)
        
        # Determine range for new value
        # Must be > current_rows[r][j-1] and < current_rows[r][j+1] (if exists)
        min_val = current_rows[r][j-1] + 1
        max_val = current_rows[r][k_plus_1 - 1] # Can't exceed current max? 
        # Actually, we can try to reduce.
        # If we want to reduce scope, we should try values < current_rows[r][j].
        # But sometimes we need to increase to move elsewhere.
        
        # Strategy: Try to reduce the value if possible.
        # If j == k, try current_rows[r][k] - 1, - 2, etc.
        # If j < k, try random in [min_val, max_val].
        
        # Let's just pick a random valid value in a window around current value
        curr_val = current_rows[r][j]
        
        # Window size
        window = 10
        start_v = max(min_val, curr_val - window)
        end_v = min(max_val, curr_val + window)
        
        if start_v > end_v:
            T *= cooling_rate
            steps += 1
            continue
            
        new_val = rng_sa.randint(start_v, end_v)
        
        if new_val == curr_val:
            continue
            
        # Check validity
        new_row, new_mask = get_new_row_diffs(current_rows[r], j, new_val)
        if new_row is None:
            continue
            
        # Check conflict with other rows
        # other_mask = global_mask & ~row_masks[r]
        other_mask = global_mask ^ row_masks[r] # Since row_masks[r] is subset of global_mask
        
        if (new_mask & other_mask) != 0:
            continue
            
        # Move is valid. Calculate delta scope
        new_scope = get_scope([row if i != r else new_row for i, row in enumerate(current_rows)])
        
        delta = new_scope - current_scope
        
        if delta < 0 or rng_sa.random() < math.exp(-delta / T):
            # Accept
            # Update state
            old_mask = row_masks[r]
            global_mask = (global_mask & ~old_mask) | new_mask
            row_masks[r] = new_mask
            current_rows[r] = new_row
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [row[:] for row in current_rows]
                
        T *= cooling_rate
        steps += 1
        
    return best_rows
