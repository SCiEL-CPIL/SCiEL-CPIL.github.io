import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, gap_factor=10):
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

def _build_static_diffs(row, k):
    """
    For each position pi (1..k), compute the set of differences in the row
    that do NOT involve position pi.
    """
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
    """
    Given a row with position pi replaced by new_val, compute the full diff set
    using the precomputed static diffs plus the k diffs involving pi.
    Returns (new_diffs_set, valid_internal).
    """
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

def _find_valid_value(row, pi, all_used, old_diffs, k, max_search=200):
    """
    Find a valid value for position pi in row that doesn't conflict with all_used
    (excluding old_diffs) and maintains internal uniqueness.
    Returns (new_val, new_diffs) or (None, None).
    """
    old_val = row[pi]
    min_allowed = row[pi - 1] + 1
    max_allowed = (row[pi + 1] - 1) if pi < k else old_val + max_search
    
    # First try decreasing values
    for new_val in range(old_val - 1, min_allowed - 1, -1):
        # Compute new diffs
        new_diffs = set()
        valid = True
        for i in range(len(row)):
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
        
        if not valid:
            continue
        
        # Check conflicts
        conflict = False
        for d in new_diffs:
            if d in all_used and d not in old_diffs:
                conflict = True
                break
        
        if not conflict:
            return new_val, new_diffs
    
    # Try increasing values
    for new_val in range(old_val + 1, max_allowed + 1):
        if pi < k and new_val >= row[pi + 1]:
            break
        
        new_diffs = set()
        valid = True
        for i in range(len(row)):
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
        
        if not valid:
            continue
        
        conflict = False
        for d in new_diffs:
            if d in all_used and d not in old_diffs:
                conflict = True
                break
        
        if not conflict:
            return new_val, new_diffs
    
    return None, None

def _row_replacement_search(rows, all_used, k, time_limit):
    """
    Try to replace each row with a better one using DFS-based search.
    Returns improved rows if found.
    """
    start_time = time.time()
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    for ri in range(len(rows)):
        if time.time() > time_limit:
            break
        
        old_row = rows[ri]
        old_diffs = _row_diffs_set(old_row)
        other_diffs = set(all_used) - old_diffs
        
        # Try to find a better row using DFS
        best_new_row = None
        best_new_max = old_row[-1]
        
        def dfs_row(row, current_max):
            nonlocal best_new_row, best_new_max
            
            if time.time() > time_limit:
                return
            
            if len(row) == k + 1:
                if current_max < best_new_max:
                    best_new_max = current_max
                    best_new_row = row[:]
                return
            
            last = row[-1]
            existing_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    existing_diffs.add(row[j] - row[i])
            
            # Try values from last+1 up to best_new_max - 1
            max_try = best_new_max - 1 if best_new_max < float('inf') else last + 50
            for x in range(last + 1, max_try + 1):
                new_diffs = {x - a for a in row}
                
                # Check against other rows' diffs
                if new_diffs & other_diffs:
                    continue
                
                # Check internal uniqueness
                if new_diffs & existing_diffs:
                    continue
                
                row.append(x)
                dfs_row(row, x)
                row.pop()
                
                if best_new_row is not None and x >= best_new_max:
                    break
        
        dfs_row([0], 0)
        
        if best_new_row is not None and best_new_max < old_row[-1]:
            # Update rows and all_used
            rows[ri] = best_new_row
            all_used = set(all_used) - old_diffs
            all_used |= _row_diffs_set(best_new_row)
            current_val = _global_max(rows)
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in rows]
    
    return best_rows, all_used

def _try_multi_pos_move(current_rows, ri, pi1, pi2, new_val1, new_val2, all_used, row_diffs, k):
    """
    Try a simultaneous move on positions pi1 and pi2 of row ri.
    Returns (new_diffs, valid) if the move is valid, else (None, False).
    """
    row = current_rows[ri]
    n = k + 1
    
    # Build the set of diffs that don't involve pi1 or pi2
    static_set = set()
    for i in range(n):
        if i == pi1 or i == pi2:
            continue
        for j in range(i + 1, n):
            if j == pi1 or j == pi2:
                continue
            static_set.add(row[j] - row[i])
    
    # Add diffs involving pi1 or pi2 (but not both)
    # pi1 with all others except pi2
    for i in range(n):
        if i == pi1 or i == pi2:
            continue
        if i < pi1:
            d = new_val1 - row[i]
        else:
            d = row[i] - new_val1
        if d in static_set:
            return None, False
        static_set.add(d)
    
    # pi2 with all others except pi1
    for i in range(n):
        if i == pi1 or i == pi2:
            continue
        if i < pi2:
            d = new_val2 - row[i]
        else:
            d = row[i] - new_val2
        if d in static_set:
            return None, False
        static_set.add(d)
    
    # pi1 with pi2
    if pi1 < pi2:
        d = new_val2 - new_val1
    else:
        d = new_val1 - new_val2
    if d in static_set:
        return None, False
    static_set.add(d)
    
    # Check against all_used (excluding old diffs of this row)
    old_diffs = row_diffs[ri]
    for d in static_set:
        if d in all_used and d not in old_diffs:
            return None, False
    
    return static_set, True

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    # Track used differences for fast checking
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Row Replacement Phase (inspired by DFS-based row search)
    replacement_deadline = min(deadline, start_time + 25.0)
    while time.time() < replacement_deadline:
        best_rows, all_used = _row_replacement_search(
            [r[:] for r in best_rows], 
            _compute_all_used(best_rows), 
            k, 
            min(replacement_deadline, time.time() + 8.0)
        )
        new_val = _global_max(best_rows)
        if new_val < best_val:
            best_val = new_val
            row_maxes = [max(r) for r in best_rows]
        else:
            break
    
    # 3. Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    steps = 0
    
    # Precompute row diffs and static diffs
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    sa_deadline = min(deadline, start_time + 95.0)
    while time.time() < sa_deadline and no_improve < max_no_improve:
        steps += 1
        
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        
        # Decide move type: single position (70%), multi-position last two (20%), random two (10%)
        move_type = random.random()
        
        if move_type < 0.70:
            # Single position move
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(50, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            max_increase = min(15, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
            
            if random.random() < 0.15:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
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
                
                conflict = False
                for d in new_diffs:
                    if d in all_used and d not in old_diffs:
                        conflict = True
                        break
                
                if conflict:
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
                    if random.random() < 0.2:
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
                    current_val = max(row_maxes)
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
                
        elif move_type < 0.90 and k >= 2:
            # Multi-position: adjust last two entries (positions k-1 and k)
            pi1 = k - 1
            pi2 = k
            old_v1 = current_rows[ri][pi1]
            old_v2 = current_rows[ri][pi2]
            
            min_v1 = current_rows[ri][pi1 - 1] + 1
            max_v1 = old_v2 - 1
            min_v2 = old_v1 + 1
            max_v2 = current_val + 20
            
            # Try decreasing the max entry
            moved = False
            for dv2 in range(1, min(30, old_v2 - min_v2) + 1):
                new_v2 = old_v2 - dv2
                for dv1 in range(-5, min(10, max_v1 - old_v1) + 1):
                    new_v1 = old_v1 + dv1
                    if new_v1 <= min_v1 or new_v1 >= new_v2:
                        continue
                    
                    new_diffs, valid = _try_multi_pos_move(
                        current_rows, ri, pi1, pi2, new_v1, new_v2, all_used, row_diffs, k)
                    
                    if not valid:
                        continue
                    
                    old_diffs = row_diffs[ri]
                    cand_row_max = new_v2
                    
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
                        current_rows[ri][pi1] = new_v1
                        current_rows[ri][pi2] = new_v2
                        row_diffs[ri] = new_diffs
                        row_maxes[ri] = cand_row_max
                        current_val = max(row_maxes)
                        static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                        
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        moved = True
                        break
                if moved:
                    break
            
            if not moved:
                no_improve += 1
        else:
            # Random two-position move
            if k < 2:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            pi1, pi2 = sorted(random.sample(range(1, k + 1), 2))
            old_v1 = current_rows[ri][pi1]
            old_v2 = current_rows[ri][pi2]
            
            min_v1 = current_rows[ri][pi1 - 1] + 1
            max_v1 = (current_rows[ri][pi1 + 1] - 1) if pi1 < k else current_val + 10
            min_v2 = current_rows[ri][pi2 - 1] + 1
            max_v2 = (current_rows[ri][pi2 + 1] - 1) if pi2 < k else current_val + 10
            
            moved = False
            for _ in range(5):
                new_v1 = random.randint(min_v1, max_v1)
                new_v2 = random.randint(min_v2, max_v2)
                if pi1 < pi2 and new_v1 >= new_v2:
                    continue
                if pi2 < pi1 and new_v2 >= new_v1:
                    continue
                
                new_diffs, valid = _try_multi_pos_move(
                    current_rows, ri, pi1, pi2, new_v1, new_v2, all_used, row_diffs, k)
                
                if not valid:
                    continue
                
                old_diffs = row_diffs[ri]
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
                    if random.random() < 0.1:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= old_diffs
                    all_used |= new_diffs
                    current_rows[ri][pi1] = new_v1
                    current_rows[ri][pi2] = new_v2
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
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
        
        # Periodic restart if stuck
        if no_improve > 15000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # 4. Final polishing phase with tighter tolerance
    polish_deadline = min(deadline, start_time + 112.0)
    T = 1.0
    T_min = 0.1
    cool = 0.99999
    no_improve = 0
    max_no_improve_polish = 10000
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Pick a row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try decreasing moves in polish phase
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(20, old_val - min_allowed)
        
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
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
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
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
                if random.random() < 0.1:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.05))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
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
    
    # 5. Final squeezing: try to reduce the max entry of the row that has the global max
    squeeze_deadline = min(deadline, start_time + 115.0)
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < squeeze_deadline:
        # Find the row with the maximum entry
        max_idx = max(range(n_rows), key=lambda i: row_maxes[i])
        max_row = current_rows[max_idx]
        max_entry = max_row[k]
        
        # Try to reduce the last entry
        moved = False
        for new_val in range(max_entry - 1, max_row[k-1], -1):
            new_diffs = set()
            valid = True
            for i in range(k):
                d = new_val - max_row[i]
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
            
            old_diffs = row_diffs[max_idx]
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Accept
            all_used -= old_diffs
            all_used |= new_diffs
            current_rows[max_idx][k] = new_val
            row_diffs[max_idx] = new_diffs
            row_maxes[max_idx] = new_val
            current_val = max(row_maxes)
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
            moved = True
            break
        
        if not moved:
            # Try reducing other positions in the max row to free up space
            for pi in range(1, k):
                old_val = current_rows[max_idx][pi]
                min_allowed = current_rows[max_idx][pi - 1] + 1
                for d in range(1, min(10, old_val - min_allowed) + 1):
                    new_val = old_val - d
                    if new_val <= min_allowed:
                        continue
                    if pi < k and new_val >= current_rows[max_idx][pi + 1]:
                        continue
                    
                    new_diffs = set()
                    valid = True
                    for i in range(k + 1):
                        if i == pi:
                            continue
                        if i < pi:
                            d2 = new_val - current_rows[max_idx][i]
                        else:
                            d2 = current_rows[max_idx][i] - new_val
                        if d2 in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d2)
                    if not valid:
                        continue
                    
                    old_diffs = row_diffs[max_idx]
                    conflict = False
                    for d2 in new_diffs:
                        if d2 in all_used and d2 not in old_diffs:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    all_used -= old_diffs
                    all_used |= new_diffs
                    current_rows[max_idx][pi] = new_val
                    row_diffs[max_idx] = new_diffs
                    row_maxes[max_idx] = current_rows[max_idx][k]
                    current_val = max(row_maxes)
                    moved = True
                    break
                if moved:
                    break
        
        if not moved:
            break  # No more improvements possible
    
    return best_rows

# EVOLVE-BLOCK-END
