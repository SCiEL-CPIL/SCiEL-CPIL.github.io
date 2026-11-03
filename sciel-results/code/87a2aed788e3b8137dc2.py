import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
        
    def random_greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find valid x
                valid_xs = []
                # Look ahead a bit
                limit = max(x + 20, len(row) * 5) 
                for candidate in range(x + 1, limit + 1):
                    new = {candidate - a for a in row}
                    if not new & used:
                        valid_xs.append(candidate)
                        if len(valid_xs) > 5: # Don't search too far
                            break
                if not valid_xs:
                    # Fallback to deterministic search
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        valid_xs = [x]
                    else:
                        # This shouldn't happen if limit is large enough, but just in case
                        continue
                
                x = random.choice(valid_xs)
                row.append(x)
                used |= {x - a for a in row[:-1]} # Update used
            rows.append(row)
        return rows, used
        
    # Try a few random inits
    best_initial_rows = None
    best_initial_max = float('inf')
    
    for _ in range(5):
        rows, used_diffs = random_greedy_init()
        m = max(max(row) for row in rows)
        if m < best_initial_max:
            best_initial_max = m
            best_initial_rows = rows
            best_initial_used = used_diffs
            
    rows = [r[:] for r in best_initial_rows]
    used_diffs = set(best_initial_used)
    current_max = best_initial_max
    best_rows = [r[:] for r in rows]
    best_max = current_max
    current_diffs_per_row = [get_diffs(row) for row in rows]
    
    # SA
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        r = random.randint(0, n_rows - 1)
        i = random.randint(1, k)
        
        old_row = rows[r][:]
        old_val = old_row[i]
        
        low = old_row[i-1] + 1
        high = old_row[i+1] - 1 if i < k else old_val + 10
        
        if low > high:
            T *= cooling
            continue
            
        upper_bound = min(old_val + 2, high)
        if upper_bound < low:
            upper_bound = low
            
        new_val = random.randint(low, upper_bound)
        
        if new_val == old_val:
            T *= cooling
            continue
            
        new_row = old_row[:]
        new_row[i] = new_val
        new_diffs_row = get_diffs(new_row)
        old_diffs_row = current_diffs_per_row[r]
        
        valid = True
        for d in new_diffs_row:
            if d in used_diffs and d not in old_diffs_row:
                valid = False
                break
                
        if not valid:
            T *= cooling
            continue
            
        used_diffs -= old_diffs_row
        used_diffs |= new_diffs_row
        rows[r] = new_row
        current_diffs_per_row[r] = new_diffs_row
        
        new_max = max(max(row) for row in rows)
        
        if new_max <= current_max:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_rows = [row[:] for row in rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            delta = new_max - current_max
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_max = new_max
                no_improve_count += 1
            else:
                used_diffs -= new_diffs_row
                used_diffs |= old_diffs_row
                rows[r] = old_row
                current_diffs_per_row[r] = old_diffs_row
                no_improve_count += 1
                
        T *= cooling
        
        if no_improve_count > max_no_improve:
            T = 5.0
            no_improve_count = 0
            
    return best_rows
