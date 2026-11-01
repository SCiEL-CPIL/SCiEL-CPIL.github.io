import time
import random
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initial greedy construction to get a valid starting point
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
    best_scope = max(max(r) for r in best_rows)
    
    # Helper to rebuild used_diffs from rows
    def build_used_diffs(rws):
        ud = set()
        for r in rws:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    ud.add(r[j] - r[i])
        return ud

    current_rows = [r[:] for r in rows]
    current_scope = best_scope
    used_diffs = build_used_diffs(current_rows)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    no_improve_steps = 0
    max_no_improve = 2000
    
    while time.time() - start_time < deadline:
        # Check for restart
        if no_improve_steps > max_no_improve:
            # Reset to best
            current_rows = [r[:] for r in best_rows]
            current_scope = best_scope
            used_diffs = build_used_diffs(current_rows)
            T = 5.0
            no_improve_steps = 0
        
        # Pick move
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        pos_idx = random.randint(1, len(row) - 1)
        
        old_val = row[pos_idx]
        old_diffs = set()
        for j in range(len(row)):
            if j != pos_idx:
                old_diffs.add(abs(old_val - row[j]))
        
        used_diffs.difference_update(old_diffs)
        
        lower_bound = row[pos_idx-1] + 1
        upper_bound = row[pos_idx+1] - 1 if pos_idx < len(row)-1 else current_scope + 50
        
        new_val = None
        
        # Try shrink
        if old_val > lower_bound:
            range_size = old_val - lower_bound
            if range_size <= 100:
                for cand in range(lower_bound, old_val):
                    valid = True
                    for j in range(len(row)):
                        if j != pos_idx:
                            d = abs(cand - row[j])
                            if d in used_diffs:
                                valid = False
                                break
                    if valid:
                        new_val = cand
                        break
            else:
                for _ in range(10):
                    cand = random.randint(lower_bound, old_val - 1)
                    valid = True
                    for j in range(len(row)):
                        if j != pos_idx:
                            d = abs(cand - row[j])
                            if d in used_diffs:
                                valid = False
                                break
                    if valid:
                        new_val = cand
                        break
        
        # Try explore
        if new_val is None:
            max_jump = int(T * 2) + 10
            search_high = min(upper_bound, old_val + max_jump)
            if search_high > old_val:
                for _ in range(10):
                    cand = random.randint(old_val + 1, search_high)
                    valid = True
                    for j in range(len(row)):
                        if j != pos_idx:
                            d = abs(cand - row[j])
                            if d in used_diffs:
                                valid = False
                                break
                    if valid:
                        new_val = cand
                        break
        
        if new_val is None:
            used_diffs.update(old_diffs)
            no_improve_steps += 1
            continue
        
        row[pos_idx] = new_val
        new_scope = max(max(r) for r in current_rows)
        delta_E = new_scope - current_scope
        
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            new_diffs = set()
            for j in range(len(row)):
                if j != pos_idx:
                    new_diffs.add(abs(new_val - row[j]))
            used_diffs.update(new_diffs)
            
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            row[pos_idx] = old_val
            used_diffs.update(old_diffs)
            no_improve_steps += 1
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_rows
