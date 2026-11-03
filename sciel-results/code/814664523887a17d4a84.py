import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    best_max = float('inf')
    best_rows = None

    # --- Helper Functions ---
    
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_max(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    # --- Greedy Baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = compute_max(best_rows)

    # --- Simulated Annealing with Improved Moves ---
    
    def sa_optimize(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len
        
        # Precompute diffs per row and global counts
        row_diffs = [get_diffs(r) for r in rows]
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        # Calculate initial collision count
        collision_count = 0
        for d, c in diff_counts.items():
            if c > 1:
                collision_count += c - 1
                
        current_max = compute_max(rows)
        
        def get_cost():
            return collision_count * 10000 + current_max

        current_cost = get_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max

        T = 50.0
        cooling = 0.9997
        min_T = 0.5
        start_time = time.time()
        
        W_COLL = 10000
        
        while time.time() - start_time < time_limit:
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.5:
                # Single element move
                r_idx = random.randint(0, n - 1)
                i_idx = random.randint(1, L - 1)
                
                row = rows[r_idx]
                old_val = row[i_idx]
                
                lower = row[i_idx - 1] + 1
                if i_idx < L - 1:
                    upper = row[i_idx + 1] - 1
                else:
                    upper = current_max + 10
                
                if lower > upper:
                    continue
                    
                # Bias towards lower values
                if random.random() < 0.7:
                    new_val = random.randint(lower, min(upper, old_val)) if old_val > lower else random.randint(lower, upper)
                else:
                    new_val = random.randint(lower, upper)
                
                if new_val == old_val:
                    continue
                
                # --- Update State ---
                old_diffs = row_diffs[r_idx]
                for d in old_diffs:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                        if c >= 2:
                            collision_count -= 1
                
                row[i_idx] = new_val
                new_diffs = get_diffs(row)
                row_diffs[r_idx] = new_diffs
                
                for d in new_diffs:
                    c = diff_counts.get(d, 0)
                    if c == 0:
                        diff_counts[d] = 1
                    else:
                        diff_counts[d] = c + 1
                        if c >= 1:
                            collision_count += 1
                
                new_global_max = compute_max(rows)
                new_cost = collision_count * W_COLL + new_global_max
                
                delta = new_cost - current_cost
                accept = False
                
                if delta < 0:
                    accept = True
                else:
                    if T > min_T:
                        p = math.exp(-delta / T)
                        if random.random() < p:
                            accept = True
                    elif delta == 0:
                        if random.random() < 0.5:
                            accept = True
                
                if accept:
                    current_cost = new_cost
                    current_max = new_global_max
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_rows = [r[:] for r in rows]
                        best_local_max = new_global_max
                else:
                    row[i_idx] = old_val
                    row_diffs[r_idx] = old_diffs
                    
                    for d in old_diffs:
                        c = diff_counts.get(d, 0)
                        if c == 0:
                            diff_counts[d] = 1
                        else:
                            diff_counts[d] = c + 1
                            if c >= 1:
                                collision_count += 1
                            
                    for d in new_diffs:
                        c = diff_counts[d]
                        if c == 1:
                            del diff_counts[d]
                        else:
                            diff_counts[d] = c - 1
                            if c >= 2:
                                collision_count -= 1
                            
                    current_max = compute_max(rows)
                    
            elif move_type < 0.8:
                # Suffix shift move: change a[i] and shift a[i+1:] by delta
                r_idx = random.randint(0, n - 1)
                i_idx = random.randint(1, L - 2)
                
                row = rows[r_idx]
                
                # Try to shift the suffix by a small amount
                delta_shift = random.choice([-2, -1, 1, 2])
                
                # Check if shift is valid
                new_suffix = [row[j] + delta_shift for j in range(i_idx, L)]
                
                # Check increasing order
                valid = True
                if i_idx > 0:
                    if new_suffix[0] <= row[i_idx - 1]:
                        valid = False
                for j in range(len(new_suffix) - 1):
                    if new_suffix[j + 1] <= new_suffix[j]:
                        valid = False
                
                if not valid:
                    continue
                
                # Check for conflicts
                old_diffs = row_diffs[r_idx]
                for d in old_diffs:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                        if c >= 2:
                            collision_count -= 1
                
                new_row = row[:i_idx] + new_suffix
                new_diffs = get_diffs(new_row)
                
                conflict_check = True
                for d in new_diffs:
                    if diff_counts.get(d, 0) > 0:
                        conflict_check = False
                        break
                
                if conflict_check:
                    row[:] = new_row
                    row_diffs[r_idx] = new_diffs
                    
                    for d in new_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    
                    new_global_max = compute_max(rows)
                    new_cost = collision_count * W_COLL + new_global_max
                    
                    delta = new_cost - current_cost
                    accept = False
                    
                    if delta < 0:
                        accept = True
                    else:
                        if T > min_T:
                            p = math.exp(-delta / T)
                            if random.random() < p:
                                accept = True
                        
                    if accept:
                        current_cost = new_cost
                        current_max = new_global_max
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_rows = [r[:] for r in rows]
                            best_local_max = new_global_max
                    else:
                        # Revert
                        row[:] = [row[j] - delta_shift for j in range(i_idx, L)]
                        row_diffs[r_idx] = old_diffs
                        
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        
                        current_max = compute_max(rows)
                else:
                    # Revert counts
                    for d in old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                        
            else:
                # Row swap move: swap two rows
                r1 = random.randint(0, n - 1)
                r2 = random.randint(0, n - 1)
                if r1 == r2:
                    continue
                
                rows[r1], rows[r2] = rows[r2], rows[r1]
                row_diffs[r1], row_diffs[r2] = row_diffs[r2], row_diffs[r1]
                
                new_global_max = compute_max(rows)
                new_cost = collision_count * W_COLL + new_global_max
                
                delta = new_cost - current_cost
                accept = False
                
                if delta < 0:
                    accept = True
                else:
                    if T > min_T:
                        p = math.exp(-delta / T)
                        if random.random() < p:
                            accept = True
                
                if accept:
                    current_cost = new_cost
                    current_max = new_global_max
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_rows = [r[:] for r in rows]
                        best_local_max = new_global_max
                else:
                    rows[r1], rows[r2] = rows[r2], rows[r1]
                    row_diffs[r1], row_diffs[r2] = row_diffs[r2], row_diffs[r1]
                    current_max = compute_max(rows)
            
            T *= cooling
            if T < min_T:
                T = min_T

        return best_local_rows, best_local_max

    # --- Polish Phase ---
    def polish(rows, time_limit):
        rows = [r[:] for r in rows]
        current_max = compute_max(rows)
        start_time = time.time()
        
        diff_counts = {}
        for r in rows:
            for d in get_diffs(r):
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        for d, c in diff_counts.items():
            if c > 1:
                return rows, current_max
                
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                # Try to reduce last element
                for new_val in range(row[-1] - 1, row[-2], -1):
                    old_last = row[-1]
                    old_diffs_to_remove = [old_last - x for x in row[:-1]]
                    new_diffs_to_add = [new_val - x for x in row[:-1]]
                    
                    valid = True
                    for d in old_diffs_to_remove:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                            
                    for d in new_diffs_to_add:
                        if d in diff_counts:
                            valid = False
                            break
                            
                    if valid:
                        row[-1] = new_val
                        for d in new_diffs_to_add:
                            diff_counts[d] = 1
                        improved = True
                        current_max = compute_max(rows)
                        break
                    else:
                        for d in old_diffs_to_remove:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        
                # Try to reduce second-to-last element
                if len(row) > 2:
                    for new_val in range(row[-2] - 1, row[-3], -1):
                        old_val = row[-2]
                        old_diffs_to_remove = [row[j] - old_val for j in range(-2, 0)] + [old_val - row[i] for i in range(0, -3, -1)]
                        new_diffs_to_add = [row[j] - new_val for j in range(-2, 0)] + [new_val - row[i] for i in range(0, -3, -1)]
                        
                        valid = True
                        for d in old_diffs_to_remove:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                                
                        for d in new_diffs_to_add:
                            if d in diff_counts:
                                valid = False
                                break
                                
                        if valid:
                            row[-2] = new_val
                            for d in new_diffs_to_add:
                                diff_counts[d] = 1
                            improved = True
                            current_max = compute_max(rows)
                            break
                        else:
                            for d in old_diffs_to_remove:
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                        
        return rows, current_max

    # Main Loop
    sa_time = min(105.0, deadline - time.time())
    if sa_time > 0:
        local_rows, local_max = sa_optimize(best_rows, sa_time)
        
        if validate(local_rows):
            if local_max < best_max:
                best_rows = local_rows
                best_max = local_max
    
    if time.time() < deadline - 1.0:
        polish_time = deadline - time.time() - 1.0
        polished_rows, polished_max = polish(best_rows, polish_time)
        if validate(polished_rows):
            if polished_max < best_max:
                best_rows = polished_rows
                best_max = polished_max

    return best_rows
