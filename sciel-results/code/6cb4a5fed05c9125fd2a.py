import time
import random

def construct(n_rows, k):
    start_time = time.time()
    # 110 seconds safety margin
    deadline = start_time + 110.0
    
    # 1. Greedy Initialization
    rows, used_diffs = _greedy_init(n_rows, k)
    
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in rows)
    
    current_rows = [r[:] for r in rows]
    current_used = set(used_diffs)
    current_scope = best_scope
    
    # Simulated Annealing Parameters
    T = 5.0
    T_min = 0.01
    # Estimate steps: 110s / ~1us per step? Python is slower. ~100us per step?
    # Let's assume 10,000 steps/sec. 1.1M steps.
    # Cooling: T_start=5, T_end=0.01.
    # rate = (0.01/5)^(1/1000000) ~ 0.999994
    cooling_rate = 0.999994
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 50000
    
    while time.time() < deadline:
        # Move Selection
        if random.random() < 0.8:
            # Single Element Move
            r = random.randint(0, n_rows - 1)
            i = random.randint(1, k)
            
            old_val = current_rows[r][i]
            
            # Define range
            lo = current_rows[r][i-1] + 1
            if i < k:
                hi = current_rows[r][i+1] - 1
            else:
                # Allow some growth to escape local minima, but keep it bounded
                hi = current_scope + 10 
                
            if lo > hi:
                continue
            
            # Random selection, biased towards smaller values if possible?
            # Just uniform for now.
            new_val = random.randint(lo, hi)
            
            if new_val == old_val:
                continue
            
            # Calculate Old Diffs
            old_diffs = set()
            for j in range(k + 1):
                if j != i:
                    old_diffs.add(abs(current_rows[r][j] - old_val))
            
            # Calculate New Diffs
            new_diffs = set()
            for j in range(k + 1):
                if j != i:
                    new_diffs.add(abs(current_rows[r][j] - new_val))
            
            # Check Conflict
            # The new diffs must not conflict with any existing diff that is NOT being removed
            remaining_used = current_used - old_diffs
            if new_diffs & remaining_used:
                continue
            
            # Apply Move
            current_used -= old_diffs
            current_used |= new_diffs
            current_rows[r][i] = new_val
            
            new_scope = max(max(row) for row in current_rows)
            
            # Acceptance
            delta = new_scope - current_scope
            if delta <= 0 or random.random() < pow(2.718, -delta / T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [row[:] for row in current_rows]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Reject
                current_used -= new_diffs
                current_used |= old_diffs
                current_rows[r][i] = old_val
                no_improve_steps += 1
        else:
            # Swap Move
            if n_rows < 2:
                continue
            r1, r2 = random.sample(range(n_rows), 2)
            i1 = random.randint(1, k)
            i2 = random.randint(1, k)
            
            v1 = current_rows[r1][i1]
            v2 = current_rows[r2][i2]
            
            if v1 == v2:
                continue
            
            # Check bounds
            if i1 > 0 and v2 <= current_rows[r1][i1-1]: continue
            if i1 < k and v2 >= current_rows[r1][i1+1]: continue
            if i2 > 0 and v1 <= current_rows[r2][i2-1]: continue
            if i2 < k and v1 >= current_rows[r2][i2+1]: continue
            
            old_diffs_1 = set(abs(current_rows[r1][j] - v1) for j in range(k+1) if j != i1)
            old_diffs_2 = set(abs(current_rows[r2][j] - v2) for j in range(k+1) if j != i2)
            
            new_diffs_1 = set(abs(current_rows[r1][j] - v2) for j in range(k+1) if j != i1)
            new_diffs_2 = set(abs(current_rows[r2][j] - v1) for j in range(k+1) if j != i2)
            
            old_all = old_diffs_1 | old_diffs_2
            new_all = new_diffs_1 | new_diffs_2
            
            remaining_used = current_used - old_all
            if new_all & remaining_used:
                continue
            
            current_used -= old_all
            current_used |= new_all
            current_rows[r1][i1] = v2
            current_rows[r2][i2] = v1
            
            new_scope = max(max(row) for row in current_rows)
            
            delta = new_scope - current_scope
            if delta <= 0 or random.random() < pow(2.718, -delta / T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [row[:] for row in current_rows]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                current_used -= new_all
                current_used |= old_all
                current_rows[r1][i1] = v1
                current_rows[r2][i2] = v2
                no_improve_steps += 1
        
        steps += 1
        if steps % 1000 == 0:
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # Reheat if stuck
            if no_improve_steps > max_no_improve:
                T = 5.0
                no_improve_steps = 0
                max_no_improve += 50000

    return best_rows

def _greedy_init(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows, used
