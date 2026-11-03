import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for safety
    
    def get_diffs(row):
        diffs = set()
        # row is list of k+1 ints
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - ri)
        return diffs

    # 1. Initial Greedy Solution
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # State for SA
    current_rows = [r[:] for r in rows]
    row_diffs = [get_diffs(r) for r in current_rows]
    global_used_diffs = set()
    for rd in row_diffs:
        global_used_diffs |= rd
        
    current_max = best_max
    T = 5.0
    T_min = 0.01
    alpha = 0.99995
    
    # Helper to update global max efficiently?
    # Just recalculate.
    
    while time.time() < deadline:
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick a random index to modify
        # Weight towards later indices? Later indices have more freedom to grow/shrink?
        # Actually, changing early indices shifts everything.
        # Let's pick uniformly.
        i_idx = random.randint(1, k)
        
        # Generate candidate value
        # Try small perturbations
        # Also try "jumps"
        if random.random() < 0.8:
            delta = random.choice([-2, -1, 1, 2])
        else:
            delta = random.choice([-10, -5, 5, 10, 20])
            
        new_val = row[i_idx] + delta
        
        # Check bounds
        if new_val <= row[i_idx - 1]:
            continue
        if i_idx < k and new_val >= row[i_idx + 1]:
            continue
        if new_val < 0:
            continue
            
        # Remove current diffs
        global_used_diffs -= row_diffs[r_idx]
        
        # Apply
        old_val = row[i_idx]
        row[i_idx] = new_val
        
        # Compute new diffs
        new_row_diffs = get_diffs(row)
        
        # Check conflict
        if new_row_diffs & global_used_diffs:
            # Revert
            row[i_idx] = old_val
            global_used_diffs |= row_diffs[r_idx]
            continue
        else:
            # Update state
            global_used_diffs |= new_row_diffs
            row_diffs[r_idx] = new_row_diffs
            
            # Calculate new global max
            new_global_max = max(max(r) for r in current_rows)
            
            # Accept/Reject
            if new_global_max < current_max:
                current_max = new_global_max
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in current_rows]
            elif new_global_max == current_max:
                if random.random() < 0.5:
                    current_max = new_global_max
                else:
                    # Revert
                    row[i_idx] = old_val
                    global_used_diffs -= new_row_diffs
                    row_diffs[r_idx] = get_diffs(row)
                    global_used_diffs |= row_diffs[r_idx]
            else:
                delta_E = new_global_max - current_max
                # Use T
                if random.random() < math.exp(-delta_E / T):
                    current_max = new_global_max
                else:
                    # Revert
                    row[i_idx] = old_val
                    global_used_diffs -= new_row_diffs
                    row_diffs[r_idx] = get_diffs(row)
                    global_used_diffs |= row_diffs[r_idx]
                    
        # Cool
        T *= alpha
        if T < T_min:
            T = T_min

    return best_rows
