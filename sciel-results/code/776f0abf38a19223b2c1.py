import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_max_val = max(max(r) for r in best_rows)
    
    MAX_DIFF = 1000 # Safe upper bound
    
    def sa_search(initial_rows, time_limit):
        rows = [list(r) for r in initial_rows]
        diff_counts = [0] * (MAX_DIFF + 1)
        
        # Initialize counts and conflicts
        current_conflicts = 0
        max_val = 0
        for r in rows:
            for i in range(len(r)):
                max_val = max(max_val, r[i])
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d <= MAX_DIFF:
                        diff_counts[d] += 1
                        if diff_counts[d] > 1:
                            current_conflicts += 1
        
        current_max = max_val
        current_cost = current_max * 1000 + current_conflicts
        
        best_rows_local = [list(r) for r in rows]
        best_max_local = current_max
        best_conflicts_local = current_conflicts
        
        T = 50.0
        cooling = 0.9995
        min_T = 0.5
        
        # Helper to update diff counts and conflicts
        def update_diffs(row_idx, j_idx, old_val, new_val, sign):
            # sign = 1 for adding, -1 for removing
            nonlocal current_conflicts
            # Remove old / Add new
            # We need to iterate over all i in the row
            for i in range(k + 1):
                if i == j_idx: continue
                if i < j_idx:
                    # diff = val_j - val_i
                    old_d = old_val - rows[row_idx][i]
                    new_d = new_val - rows[row_idx][i]
                else:
                    # diff = val_i - val_j
                    old_d = rows[row_idx][i] - old_val
                    new_d = rows[row_idx][i] - new_val
                
                if old_d <= MAX_DIFF:
                    old_count = diff_counts[old_d]
                    diff_counts[old_d] -= sign # Wait, sign is for the operation on the state
                    # If we are REMOVING old_val, we subtract from count.
                    # If we are ADDING new_val, we add to count.
                    # Let's just do explicit remove and add.
            
            # Explicit Remove Old
            for i in range(k + 1):
                if i == j_idx: continue
                if i < j_idx:
                    d = old_val - rows[row_idx][i]
                else:
                    d = rows[row_idx][i] - old_val
                if d <= MAX_DIFF:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 1:
                        current_conflicts -= 1
            
            # Explicit Add New
            for i in range(k + 1):
                if i == j_idx: continue
                if i < j_idx:
                    d = new_val - rows[row_idx][i]
                else:
                    d = rows[row_idx][i] - new_val
                if d <= MAX_DIFF:
                    diff_counts[d] += 1
                    if diff_counts[d] == 2:
                        current_conflicts += 1

        while time.time() < time_limit and T > min_T:
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k)
            
            old_val = rows[r_idx][j_idx]
            
            low = rows[r_idx][j_idx - 1] + 1
            high = rows[r_idx][j_idx + 1] - 1 if j_idx < k else best_max_local + 20
            high = min(high, best_max_local + 50)
            
            if low > high:
                T *= cooling
                continue
                
            new_val = random.randint(low, high)
            if new_val == old_val:
                T *= cooling
                continue
            
            # Save old state for revert
            rows[r_idx][j_idx] = new_val
            update_diffs(r_idx, j_idx, old_val, new_val, 1)
            
            new_max = max(max(r) for r in rows)
            new_cost = new_max * 1000 + current_conflicts
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                current_max = new_max
                current_cost = new_cost
                if current_conflicts == 0 and new_max < best_max_local:
                    best_max_local = new_max
                    best_conflicts_local = 0
                    best_rows_local = [list(r) for r in rows]
            else:
                # Revert
                rows[r_idx][j_idx] = old_val
                update_diffs(r_idx, j_idx, new_val, old_val, 1) # Note: update_diffs expects (old, new) in context of current state?
                # My update_diffs removes 'old_val' and adds 'new_val'.
                # To revert, we remove 'new_val' and add 'old_val'.
                # So call update_diffs(r_idx, j_idx, new_val, old_val, 1)
            
            T *= cooling
        
        return best_rows_local, best_max_local, best_conflicts_local

    # Run multiple SA runs if time permits
    while time.time() < deadline:
        res_rows, res_max, res_conf = sa_search(best_rows, deadline)
        if res_conf == 0 and res_max < best_max_val:
            best_max_val = res_max
            best_rows = res_rows
            if best_max_val <= 112:
                break

    return best_rows
