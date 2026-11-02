import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k):
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _build_static_diffs(row, k):
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                s.add(row[j] - row[i])
        static[pi] = s
    return static

def _compute_new_diffs_fast(row, pi, new_val, static_diffs_pi):
    n = len(row)
    new_diffs = set(static_diffs_pi)
    valid = True
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in new_diffs:
            valid = False
            break
        new_diffs.add(d)
    return new_diffs, valid

def _multi_position_move(current_rows, ri, all_used, k):
    row = current_rows[ri][:]
    row_diffs = _row_diffs_set(row)
    other_diffs = all_used - row_diffs
    original_max = row[k]
    new_max = original_max
    changed = False
    
    for pi in range(k, 0, -1):
        old_val = row[pi]
        min_allowed = row[pi - 1] + 1
        if pi < k:
            max_allowed = row[pi + 1] - 1
        else:
            max_allowed = old_val - 1
        if min_allowed >= max_allowed:
            continue
        candidates = []
        if pi == k:
            for d in range(1, min(25, old_val - min_allowed + 1)):
                candidates.append(old_val - d)
        else:
            for d in range(1, min(8, old_val - min_allowed + 1)):
                candidates.append(old_val - d)
        random.shuffle(candidates)
        for new_val in candidates:
            if new_val <= min_allowed:
                continue
            if pi < k and new_val >= max_allowed:
                continue
            new_diffs_set = set()
            valid = True
            for i in range(len(row)):
                if i == pi:
                    continue
                for j in range(i + 1, len(row)):
                    if j == pi:
                        continue
                    new_diffs_set.add(row[j] - row[i])
            for i in range(len(row)):
                if i == pi:
                    continue
                if i < pi:
                    d = new_val - row[i]
                else:
                    d = row[i] - new_val
                if d in new_diffs_set or d in other_diffs:
                    valid = False
                    break
                new_diffs_set.add(d)
            if valid and not (new_diffs_set & other_diffs):
                row[pi] = new_val
                if pi == k:
                    new_max = new_val
                changed = True
                row_diffs = new_diffs_set
                other_diffs = all_used - row_diffs
                break
    if changed and new_max < original_max:
        return True, row, row_diffs
    return False, None, None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    row_maxes = [max(r) for r in rows]
    
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    T = 150.0
    T_min = 0.3
    cool = 0.99997
    
    no_improve = 0
    max_no_improve = 25000
    
    multi_deadline = start_time + 25.0
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    while time.time() < multi_deadline and current_val > 105:
        ri = random.randrange(n_rows)
        success, new_row, new_diffs = _multi_position_move(current_rows, ri, all_used, k)
        if success:
            old_diffs = row_diffs[ri]
            all_used -= old_diffs
            all_used |= new_diffs
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            row_maxes[ri] = max(new_row)
            current_val = max(row_maxes)
            static_diffs[ri] = _build_static_diffs(new_row, k)
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if no_improve > 400:
            break
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        candidates = []
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(40, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 25
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        if random.random() < 0.15:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 15
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
                
            old_diffs = row_diffs[ri]
            other_diffs = all_used - old_diffs
            if new_diffs & other_diffs:
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.15:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 150.0
            no_improve = 0
    
    polish_deadline = min(deadline, start_time + 110.0)
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    no_improve = 0
    multi_polish_deadline = min(polish_deadline, start_time + 85.0)
    while time.time() < multi_polish_deadline and no_improve < 800:
        if max_row_indices:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        success, new_row, new_diffs = _multi_position_move(current_rows, ri, all_used, k)
        if success:
            old_diffs = row_diffs[ri]
            all_used -= old_diffs
            all_used |= new_diffs
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            row_maxes[ri] = max(new_row)
            current_val = max(row_maxes)
            static_diffs[ri] = _build_static_diffs(new_row, k)
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
                no_improve = 0
                max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
            else:
                no_improve += 1
        else:
            no_improve += 1
    
    no_improve = 0
    max_no_improve_polish = 18000
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        if max_row_indices and random.random() < 0.75:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        
        if ri in max_row_indices and random.random() < 0.55:
            pi = k
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(35, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        candidates = [old_val - d for d in range(1, max_decrease + 1)]
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
            
            old_diffs = row_diffs[ri]
            other_diffs = all_used - old_diffs
            if new_diffs & other_diffs:
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.08
            else:
                accept = False
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
    
    final_deadline = deadline
    T = 0.15
    T_min = 0.01
    cool = 0.99999
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < final_deadline:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(25, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
        
        n_try = min(max_decrease, random.randint(4, 10))
        candidates = random.sample(range(1, max_decrease + 1), n_try)
        candidates = [old_val - d for d in candidates]
        
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
            
            old_diffs = row_diffs[ri]
            other_diffs = all_used - old_diffs
            if new_diffs & other_diffs:
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.04
            else:
                accept = False
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                
                moved = True
                break
    
    return best_rows

# EVOLVE-BLOCK-END
