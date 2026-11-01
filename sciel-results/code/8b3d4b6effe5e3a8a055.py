import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 140.0
    
    def generate_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    d = x - a
                    if d in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    current_rows = generate_greedy()
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    current_diffs = [get_diffs(r) for r in current_rows]
    global_used = set()
    for d in current_diffs:
        global_used.update(d)
        
    current_scope = max(max(r) for r in current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Store max of each row for faster scope calculation
    row_maxs = [r[-1] for r in current_rows]
    
    T = 100.0
    cooling = 0.9999
    min_T = 0.1
    
    row_len = k + 1
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        diffs_r = current_diffs[r_idx]
        
        j_idx = random.randint(1, k)
        
        low = row[j_idx - 1] + 1
        high = row[j_idx + 1] - 1 if j_idx < k else current_scope + 100
        
        if random.random() < 0.7:
            # Bias towards reduction
            max_val = min(high, row[j_idx] - 1)
            if low <= max_val:
                new_val = random.randint(low, max_val)
            else:
                continue
        else:
            if low <= high:
                new_val = random.randint(low, high)
            else:
                continue
        
        if new_val == row[j_idx]:
            continue
            
        new_row = row[:]
        new_row[j_idx] = new_val
        
        # Incremental diff update
        old_involved = set()
        for i in range(j_idx):
            old_involved.add(row[j_idx] - row[i])
        for i in range(j_idx + 1, row_len):
            old_involved.add(row[i] - row[j_idx])
            
        new_involved = set()
        for i in range(j_idx):
            new_involved.add(new_row[j_idx] - new_row[i])
        for i in range(j_idx + 1, row_len):
            new_involved.add(new_row[i] - new_row[j_idx])
            
        # Check validity
        valid = True
        for d in new_involved:
            if d in global_used:
                if d not in diffs_r:
                    valid = False
                    break
        
        if not valid:
            continue
            
        # Calculate new scope
        new_max = new_row[-1]
        # Update row_maxs temporarily? No, just calc max of others
        # Since n_rows is small (4), this is fast
        other_max = 0
        for i in range(n_rows):
            if i != r_idx:
                if row_maxs[i] > other_max:
                    other_max = row_maxs[i]
                    
        new_global_scope = max(new_max, other_max)
        
        delta_scope = new_global_scope - current_scope
        
        if delta_scope <= 0 or random.random() < math.exp(-delta_scope / T):
            # Update state
            current_rows[r_idx] = new_row
            row_maxs[r_idx] = new_max
            
            # Update diffs_r
            new_diffs = diffs_r.copy()
            new_diffs.difference_update(old_involved)
            new_diffs.update(new_involved)
            current_diffs[r_idx] = new_diffs
            
            # Update global_used
            global_used.difference_update(old_involved)
            global_used.update(new_involved)
            
            current_scope = new_global_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
                
        T *= cooling
        
    return best_rows
