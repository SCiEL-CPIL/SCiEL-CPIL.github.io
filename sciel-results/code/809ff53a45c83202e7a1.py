import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct over all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Helper to get all differences for a specific row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # 1. Randomized Greedy Construction
    def greedy_randomized(n_rows, k, window_size=20, rand_prob=0.1):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                start_val = row[-1] + 1
                valid_candidates = []
                # Limit search depth to avoid infinite loops
                # We look for the smallest valid x, but sometimes pick a larger one
                # A bound on x: maybe row[-1] + k + 10?
                # Actually, just scan upwards.
                found_any = False
                for x in range(start_val, start_val + window_size * 5):
                    new = {x - a for a in row}
                    if not (new & used):
                        valid_candidates.append(x)
                        if len(valid_candidates) >= window_size:
                            break
                if not valid_candidates:
                    # Fallback to standard greedy if window search failed
                    x = start_val
                    while True:
                        new = {x - a for a in row}
                        if not (new & used):
                            valid_candidates.append(x)
                            break
                        x += 1
                
                if valid_candidates:
                    if random.random() < rand_prob and len(valid_candidates) > 1:
                        x = random.choice(valid_candidates)
                    else:
                        x = valid_candidates[0]
                    
                    row.append(x)
                    used.update({x - a for a in row[:-1]})
                else:
                    break
            rows.append(row)
        return rows, used

    # 2. Local Optimization with Simulated Annealing-like acceptance
    def local_optimize(rows, used, deadline):
        current_max = max(max(r) for r in rows)
        
        # Pre-compute diffs for each row
        row_diffs = [get_row_diffs(r) for r in rows]
        
        while time.time() < deadline:
            # Pick a random row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            
            # Pick a random position to try to reduce
            # Focus on the end of the row to reduce scope directly
            pos = random.randint(1, len(row) - 1)
            
            current_val = row[pos]
            
            lower_bound = row[pos - 1] + 1
            upper_bound = current_val - 1
            
            if pos < len(row) - 1:
                upper_bound = min(upper_bound, row[pos + 1] - 1)
                
            if lower_bound > upper_bound:
                continue
                
            # Collect old diffs involving position pos
            old_diffs = set()
            for i in range(len(row)):
                if i != pos:
                    if row[pos] > row[i]:
                        old_diffs.add(row[pos] - row[i])
                    else:
                        old_diffs.add(row[i] - row[pos])
            
            # Remove old diffs from used set temporarily
            available_diffs = used - old_diffs
            
            new_val_found = False
            # Try to find a smaller valid value
            for v in range(lower_bound, upper_bound + 1):
                new_diffs = set()
                for i in range(len(row)):
                    if i != pos:
                        if v > row[i]:
                            new_diffs.add(v - row[i])
                        else:
                            new_diffs.add(row[i] - v)
                
                if not (new_diffs & available_diffs):
                    # Valid move
                    row[pos] = v
                    used -= old_diffs
                    used |= new_diffs
                    row_diffs[r_idx] = get_row_diffs(row)
                    
                    new_max = max(max(r) for r in rows)
                    if new_max < current_max:
                        current_max = new_max
                    new_val_found = True
                    break
            
            # Occasionally try to increase a value to create slack (annealing-like)
            if not new_val_found and random.random() < 0.1:
                # Try to increase a non-last element
                if pos < len(row) - 1:
                    lower_bound2 = current_val + 1
                    upper_bound2 = row[pos + 1] - 1
                    if lower_bound2 <= upper_bound2:
                        old_diffs2 = set()
                        for i in range(len(row)):
                            if i != pos:
                                if current_val > row[i]:
                                    old_diffs2.add(current_val - row[i])
                                else:
                                    old_diffs2.add(row[i] - current_val)
                        
                        available_diffs2 = used - old_diffs2
                        for v in range(lower_bound2, upper_bound2 + 1):
                            new_diffs2 = set()
                            for i in range(len(row)):
                                if i != pos:
                                    if v > row[i]:
                                        new_diffs2.add(v - row[i])
                                    else:
                                        new_diffs2.add(row[i] - v)
                            
                            if not (new_diffs2 & available_diffs2):
                                row[pos] = v
                                used -= old_diffs2
                                used |= new_diffs2
                                row_diffs[r_idx] = get_row_diffs(row)
                                break

        return rows, current_max

    best_rows = None
    best_score = float('inf')
    best_used = None
    
    # Run multiple randomized greedy starts
    attempts = 0
    while time.time() < deadline - 10:
        window = random.randint(10, 50)
        prob = random.uniform(0.0, 0.3)
        
        rows, used = greedy_randomized(n_rows, k, window, prob)
        
        score = max(max(r) for r in rows)
        
        if score < best_score:
            best_score = score
            best_rows = [r[:] for r in rows]
            best_used = set(used)
        
        attempts += 1
        if attempts > 20 and time.time() > start_time + 30:
            break

    if best_rows is None:
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not (new & used):
                        break
                    x += 1
                row.append(x)
                used |= new
            rows.append(row)
        best_rows = rows
        best_used = used
        best_score = max(max(r) for r in rows)

    # Optimize the best solution found
    opt_rows = [r[:] for r in best_rows]
    opt_used = set(best_used)
    
    try:
        opt_rows, final_score = local_optimize(opt_rows, opt_used, deadline)
        if final_score < best_score:
            best_rows = opt_rows
            best_score = final_score
    except Exception:
        pass

    return best_rows
