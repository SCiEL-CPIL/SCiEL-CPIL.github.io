import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _compute_new_diffs(cand_row, pi, old_diffs):
    """
    Compute new diffs for a row after changing position pi.
    Only checks diffs involving position pi against the unchanged diffs.
    Returns (new_diffs_set, is_valid_internal).
    """
    n = len(cand_row)
    new_val = cand_row[pi]
    
    new_diffs = set()
    
    for i in range(n):
        if i == pi:
            continue
        for j in range(i + 1, n):
            if j == pi:
                continue
            new_diffs.add(cand_row[j] - cand_row[i])
    
    for i in range(n):
        if i == pi:
            continue
        d = abs(cand_row[pi] - cand_row[i])
        if d in new_diffs:
            return None, False
        new_diffs.add(d)
    
    return new_diffs, True

def _greedy_init(n_rows, k):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
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

def _greedy_init_with_max_limit(n_rows, k, max_limit):
    """
    Greedy initialization that tries to keep all entries <= max_limit.
    Returns None if it fails.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_limit:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _try_move(current_rows, row_diffs, all_used, ri, pi, new_val, k):
    """
    Try to move position pi in row ri to new_val.
    Returns (success, new_diffs, cand_row_max) or (False, None, None).
    """
    if new_val <= current_rows[ri][pi - 1]:
        return False, None, None
    if pi < k and new_val >= current_rows[ri][pi + 1]:
        return False, None, None
    
    cand_row = current_rows[ri][:]
    cand_row[pi] = new_val
    
    old_diffs = row_diffs[ri]
    new_diffs, valid_internal = _compute_new_diffs(cand_row, pi, old_diffs)
    
    if not valid_internal:
        return False, None, None
    
    # Check for conflicts
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return False, None, None
    
    cand_row_max = max(cand_row)
    return True, new_diffs, cand_row_max

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # 1. Multiple diverse initializations
    best_rows = None
    best_val = float('inf')
    
    # Standard greedy
    rows = _greedy_init(n_rows, k)
    if rows:
        val = _global_max(rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
    
    # Try greedy with various max limits to find tighter initial solutions
    for limit in range(100, 200, 5):
        rows_limited = _greedy_init_with_max_limit(n_rows, k, limit)
        if rows_limited:
            val = _global_max(rows_limited)
            if val < best_val:
                best_val = val
                best_rows = [r[:] for r in rows_limited]
    
    # Also try shuffled greedy: randomize the order in which we pick values
    for trial in range(5):
        rows_trial = []
        used = set()
        success = True
        for r in range(n_rows):
            row = [0]
            candidates = []
            x = 1
            while len(row) < k + 1:
                x += 1
                if x > best_val + 20:
                    success = False
                    break
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    candidates.append((x, new_diffs))
            if not success:
                break
            # Pick a random candidate from the first few valid ones
            if candidates:
                # Bias toward smaller values but allow some randomness
                idx = random.randint(0, min(4, len(candidates) - 1))
                chosen_x, chosen_diffs = candidates[idx]
                row.append(chosen_x)
                used |= chosen_diffs
            else:
                success = False
                break
        if success and rows_trial:
            val = _global_max(rows_trial)
            if val < best_val:
                best_val = val
                best_rows = [r[:] for r in rows_trial]
    
    if best_rows is None:
        best_rows = _greedy_init(n_rows, k)
        best_val = _global_max(best_rows)
    
    # 2. Multi-start local search with restarts
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    no_improve = 0
    max_no_improve = 15000
    steps = 0
    
    # Phase 1: Hill climbing with random restarts
    while time.time() < deadline and no_improve < max_no_improve:
        steps += 1
        
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(80, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        if random.random() < 0.15:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 25
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            success, new_diffs, cand_row_max = _try_move(current_rows, row_diffs, all_used, ri, pi, new_val, k)
            
            if not success:
                continue
            
            # Compute new global max
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            # Accept if improving or neutral
            if delta <= 0:
                # Update state
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = current_rows[ri][:]
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
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
        
        # Restart if stuck
        if no_improve > 8000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            no_improve = 0
    
    # 3. Targeted polish: focus on reducing the global maximum
    polish_deadline = min(deadline, start_time + 118.0)
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    no_improve = 0
    max_no_improve_polish = 30000
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Bias toward rows with the maximum entry
        if max_row_indices and random.random() < 0.8:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        
        # For the max row, prioritize the last position
        if ri in max_row_indices and random.random() < 0.6:
            pi = k
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(60, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        # Try decreasing moves, most aggressive first
        candidates = [old_val - d for d in range(max_decrease, 0, -1)]
        
        moved = False
        for new_val in candidates:
            success, new_diffs, cand_row_max = _try_move(current_rows, row_diffs, all_used, ri, pi, new_val, k)
            
            if not success:
                continue
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                # Update state
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = current_rows[ri][:]
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                    
                moved = True
                break
            elif delta == 0 and random.random() < 0.05:
                # Occasionally accept neutral moves to escape local optima
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = current_rows[ri][:]
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                no_improve += 1
                moved = True
                break
        
        if not moved:
            no_improve += 1
    
    # 4. Final fine-tuning pass
    final_deadline = deadline
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < final_deadline:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
        
        n_try = min(max_decrease, random.randint(3, 10))
        candidates = random.sample(range(1, max_decrease + 1), n_try)
        candidates = [old_val - d for d in candidates]
        
        for new_val in candidates:
            success, new_diffs, cand_row_max = _try_move(current_rows, row_diffs, all_used, ri, pi, new_val, k)
            
            if not success:
                continue
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            if new_global_max < current_val:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = current_rows[ri][:]
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                break
    
    return best_rows

# EVOLVE-BLOCK-END
