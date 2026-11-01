import random
import math
import time

def construct(n_rows, k):
    row_len = k + 1
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_init():
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        for _ in range(n_rows * k):
            # Pick row with min length, then min last val
            best_i = -1
            best_key = None
            for i in range(n_rows):
                if len(rows[i]) < row_len:
                    key = (len(rows[i]), rows[i][-1])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_i = i
            
            if best_i == -1:
                break
                
            r = rows[best_i]
            x = r[-1] + 1
            while True:
                # Check if x extends r validly
                new_diffs = {x - v for v in r}
                if not (new_diffs & used_diffs):
                    r.append(x)
                    used_diffs.update(new_diffs)
                    break
                x += 1
        return rows

    def is_valid_move(row, new_val, idx, global_diffs_except_row, row_diffs):
        # row is the current row, we are replacing row[idx] with new_val
        # global_diffs_except_row contains diffs from all other rows
        # row_diffs is the set of diffs from the original row (before change)
        
        # 1. Check increasing order
        if idx > 0 and new_val <= row[idx - 1]:
            return False
        if idx < len(row) - 1 and new_val >= row[idx + 1]:
            return False
            
        # 2. Construct new row
        new_row = row[:]
        new_row[idx] = new_val
        
        # 3. Calculate new diffs for this row
        new_row_diffs = get_row_diffs(new_row)
        
        # 4. Check internal uniqueness (always true for strictly increasing ints? 
        #    No, a_j - a_i can equal a_l - a_m. 
        #    e.g., 0, 1, 2 -> diffs 1, 2, 1. 1 appears twice.
        #    So we must check if len(new_row_diffs) == len(new_row)*(len(new_row)-1)//2
        if len(new_row_diffs) != len(new_row) * (len(new_row) - 1) // 2:
            return False
            
        # 5. Check against global diffs from other rows
        if new_row_diffs & global_diffs_except_row:
            return False
            
        return True

    start_time = time.time()
    timeout = 110
    
    # 1. Initialize
    current_rows = greedy_init()
    current_diffs = [get_row_diffs(r) for r in current_rows]
    
    # Build global set of diffs for all rows
    # We need to be able to remove a row's diffs easily.
    # Let's just keep a list of sets.
    
    def get_global_except(row_idx):
        g = set()
        for i in range(n_rows):
            if i != row_idx:
                g.update(current_diffs[i])
        return g

    current_cost = max(max(r) for r in current_rows)
    best_rows = [r[:] for r in current_rows]
    best_cost = current_cost
    best_diffs = [d.copy() for d in current_diffs]
    
    T = 50.0
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        # Pick a row and an index to modify
        r_idx = random.randint(0, n_rows - 1)
        # Prefer modifying larger values to reduce cost?
        # Or random index.
        j_idx = random.randint(1, len(current_rows[r_idx]) - 1)
        
        old_val = current_rows[r_idx][j_idx]
        
        # Try to decrease the value
        # New value must be > current_rows[r_idx][j_idx-1]
        # And < current_rows[r_idx][j_idx+1] (if exists)
        
        lower_bound = current_rows[r_idx][j_idx - 1] + 1
        upper_bound = current_rows[r_idx][j_idx + 1] - 1 if j_idx < len(current_rows[r_idx]) - 1 else old_val - 1
        
        if lower_bound > old_val:
            # Can't decrease
            continue
            
        # Try a random value in [lower_bound, old_val - 1]
        # Or just try old_val - 1, old_val - 2...
        # To be efficient, pick a random candidate
        new_val = random.randint(lower_bound, old_val - 1)
        
        # Check validity
        global_except = get_global_except(r_idx)
        
        if is_valid_move(current_rows[r_idx], new_val, j_idx, global_except, current_diffs[r_idx]):
            # Calculate new cost
            # Only r_idx changes
            new_max_in_row = max(current_rows[r_idx]) # Note: new_val < old_val, so max might decrease or stay same
            # Actually, if j_idx is the last element, max decreases.
            # If not, max stays same unless old_val was the max.
            
            # Update state
            current_rows[r_idx][j_idx] = new_val
            current_diffs[r_idx] = get_row_diffs(current_rows[r_idx])
            
            new_cost = max(max(r) for r in current_rows)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [r[:] for r in current_rows]
                    best_diffs = [d.copy() for d in current_diffs]
            else:
                # Revert
                current_rows[r_idx][j_idx] = old_val
                current_diffs[r_idx] = get_row_diffs(current_rows[r_idx])
                
        T *= cooling
        if T < 1e-6:
            break

    return best_rows
