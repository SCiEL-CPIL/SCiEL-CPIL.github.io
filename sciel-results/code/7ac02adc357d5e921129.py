import random
import math
import time

def construct(n_rows, k):
    row_len = k + 1
    
    def get_diffs(row, idx):
        val = row[idx]
        diffs = set()
        for j, v in enumerate(row):
            if j != idx:
                diffs.add(abs(val - v))
        return diffs

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                candidates = []
                x = row[-1] + 1
                limit = row[-1] + 100 
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    x = row[-1] + limit + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            candidates.append(x)
                            break
                        x += 1
                            
                candidates.sort()
                pick_range = min(5, len(candidates))
                x = random.choice(candidates[:pick_range])
                
                new_diffs = {x - a for a in row}
                row.append(x)
                used.update(new_diffs)
            rows.append(row)
        return rows, used

    def calculate_cost(rows):
        return max(max(r) for r in rows)

    best_rows_overall = None
    best_cost_overall = float('inf')
    
    start_time_global = time.time()
    max_total_time = 110
    
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() - start_time_global > max_total_time - 2:
            break
            
        rows, used = greedy_init()
        
        current_rows = [r[:] for r in rows]
        current_used = set(used)
        current_cost = calculate_cost(current_rows)
        
        best_rows = [r[:] for r in current_rows]
        best_cost = current_cost
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9998
        
        start_time = time.time()
        remaining_restarts = n_restarts - restart
        max_time = (max_total_time - (time.time() - start_time_global)) / remaining_restarts
        
        while time.time() - start_time < max_time and T > T_min:
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Local Shift
                p_idx = random.randint(1, row_len - 1)
                
                low = row[p_idx - 1] + 1
                high = row[p_idx + 1] - 1 if p_idx < row_len - 1 else current_cost + 10
                
                if high < low:
                    T *= cooling
                    continue
                    
                new_val = random.randint(low, high)
                
                if new_val == row[p_idx]:
                    T *= cooling
                    continue
                    
                old_diffs = get_diffs(row, p_idx)
                
                new_diffs = set()
                for j, v in enumerate(row):
                    if j != p_idx:
                        new_diffs.add(abs(new_val - v))
                        
                if len(new_diffs) != row_len - 1:
                    T *= cooling
                    continue
                    
                valid = True
                for d in new_diffs:
                    if d in current_used and d not in old_diffs:
                        valid = False
                        break
                        
                if not valid:
                    T *= cooling
                    continue
                    
                old_val = row[p_idx]
                row[p_idx] = new_val
                
                current_used -= old_diffs
                current_used |= new_diffs
                
                new_cost = calculate_cost(current_rows)
                
                delta_cost = new_cost - current_cost
                if delta_cost <= 0:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_rows = [r[:] for r in current_rows]
                else:
                    if random.random() < math.exp(-delta_cost / T):
                        current_cost = new_cost
                    else:
                        row[p_idx] = old_val
                        current_used -= new_diffs
                        current_used |= old_diffs
                        
            else:
                # Reinsertion Move
                p_idx = random.randint(1, row_len - 1)
                old_row = row[:]
                old_val = row[p_idx]
                old_diffs = get_diffs(row, p_idx)
                
                temp_row = row[:p_idx] + row[p_idx+1:]
                
                found = False
                new_val = -1
                new_diffs = set()
                new_insert_idx = -1
                best_candidate_max = float('inf')
                
                # Scan for best valid reinsertion
                for i in range(len(temp_row) + 1):
                    low = temp_row[i-1] + 1 if i > 0 else 1
                    # Limit high to avoid huge scans in the last gap
                    high = temp_row[i] - 1 if i < len(temp_row) else min(current_cost + 10, old_val + 20)
                    
                    if low > high:
                        continue
                        
                    for v in range(low, high + 1):
                        nd = set()
                        for j, val in enumerate(temp_row):
                            nd.add(abs(v - val))
                        
                        valid = True
                        for d in nd:
                            if d in current_used and d not in old_diffs:
                                valid = False
                                break
                            
                        if not valid:
                            continue
                            
                        row_max = max(temp_row[-1], v) if temp_row else v
                        
                        if row_max < best_candidate_max:
                            best_candidate_max = row_max
                            new_val = v
                            new_diffs = nd
                            new_insert_idx = i
                            
                if new_val != -1:
                    # Apply move
                    new_row = temp_row[:new_insert_idx] + [new_val] + temp_row[new_insert_idx:]
                    current_rows[r_idx] = new_row
                    
                    current_used -= old_diffs
                    current_used |= new_diffs
                    
                    new_cost = calculate_cost(current_rows)
                    
                    delta_cost = new_cost - current_cost
                    if delta_cost <= 0:
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_rows = [r[:] for r in current_rows]
                    else:
                        if random.random() < math.exp(-delta_cost / T):
                            current_cost = new_cost
                        else:
                            # Revert
                            current_rows[r_idx] = old_row
                            current_used -= new_diffs
                            current_used |= old_diffs
                else:
                    T *= cooling
                    continue

            T *= cooling
            
        if best_cost < best_cost_overall:
            best_cost_overall = best_cost
            best_rows_overall = [r[:] for r in best_rows]
            
    return best_rows_overall
