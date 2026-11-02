import time
import random
import math

# EVOLVE-BLOCK-START

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

def _try_decrease_row(row, all_used, k):
    """
    Try to reduce the maximum entry of a single row by moving positions right-to-left.
    Returns (new_row, new_diffs, success) where success indicates if any change was made.
    """
    n = k + 1
    row_diffs = _row_diffs_set(row)
    other_diffs = all_used - row_diffs
    changed = False
    
    # Work from right to left to minimize conflicts
    for pi in range(k, 0, -1):
        old_val = row[pi]
        min_allowed = row[pi - 1] + 1
        if pi < k:
            max_allowed = row[pi + 1] - 1
        else:
            max_allowed = old_val - 1
        
        if min_allowed >= max_allowed:
            continue
            
        # Try decreasing values from just below old_val down to min_allowed
        for new_val in range(old_val - 1, min_allowed - 1, -1):
            # Check if this value works with the current row state
            # Compute new diffs for this position
            new_diffs_set = set()
            valid = True
            
            # Diffs not involving position pi
            for i in range(n):
                if i == pi:
                    continue
                for j in range(i + 1, n):
                    if j == pi:
                        continue
                    new_diffs_set.add(row[j] - row[i])
            
            # Diffs involving position pi
            for i in range(n):
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
            
            if valid:
                # Check that new_diffs_set doesn't conflict with other_diffs
                if not (new_diffs_set & other_diffs):
                    row[pi] = new_val
                    changed = True
                    # Update row_diffs for subsequent positions
                    row_diffs = new_diffs_set
                    other_diffs = all_used - row_diffs
                    break
    
    if changed:
        final_diffs = _row_diffs_set(row)
        return row, final_diffs, True
    else:
        return row, row_diffs, False

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
    that do NOT involve position pi. These are differences between pairs
    (i,j) where i != pi and j != pi.
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
    using the precomputed static diffs (diffs not involving pi) plus the k diffs
    involving pi. Returns (new_diffs_set, valid_internal).
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

def _multi_position_move(current_rows, ri, all_used, k):
    """
    Try to reduce multiple positions in row ri simultaneously.
    Returns (success, new_row, new_diffs).
    """
    row = current_rows[ri][:]
    row_diffs = _row_diffs_set(row)
    other_diffs = all_used - row_diffs
    
    original_max = row[k]
    new_max = original_max
    changed = False
    
    # Try to reduce from right to left
    for pi in range(k, 0, -1):
        old_val = row[pi]
        min_allowed = row[pi - 1] + 1
        if pi < k:
            max_allowed = row[pi + 1] - 1
        else:
            max_allowed = old_val - 1
            
        if min_allowed >= max_allowed:
            continue
        
        # Try a few candidate values
        candidates = []
        # Try values that would reduce the max if pi == k
        if pi == k:
            for d in range(1, min(20, old_val - min_allowed + 1)):
                candidates.append(old_val - d)
        else:
            # For non-max positions, try small adjustments
            for d in range(1, min(5, old_val - min_allowed + 1)):
                candidates.append(old_val - d)
        
        random.shuffle(candidates)
        
        for new_val in candidates:
            if new_val <= min_allowed:
                continue
            if pi < k and new_val >= max_allowed:
                continue
                
            # Check validity
            new_diffs_set = set()
            valid = True
            
            # Diffs not involving pi
            for i in range(len(row)):
                if i == pi:
                    continue
                for j in range(i + 1, len(row)):
                    if j == pi:
                        continue
                    new_diffs_set.add(row[j] - row[i])
            
            # Diffs involving pi
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
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    # Track used differences for fast checking
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Simulated Annealing
    current_rows = [r[:] for r in rows]
    current_val = best_val
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    
    # Phase 1: Multi-position moves to quickly reduce scope
    multi_deadline = start_time + 30.0
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
        if no_improve > 500:
            break
    
    # Precompute row diffs and static diffs
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates - try more of these, but limit to recent range
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        # Bias toward smaller decreases first (more likely to be valid)
        for d in range(1, max_decrease + 1):
            new_v = old_val - d
            candidates.append(new_v)
            
        # Increasing candidates (for exploration) - fewer
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(10, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            new_v = old_val + d
            candidates.append(new_v)
            
        # Random jump (large perturbation)
        if random.random() < 0.1:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        # Pre-fetch static diffs for this position
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            # Basic validity check for this position
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Fast diff computation using static diffs
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
                
            # Get old diffs of row ri
            old_diffs = row_diffs[ri]
            
            # Check for conflicts: new_diffs should not intersect with (all_used - old_diffs)
            # Use set intersection for speed
            other_diffs = all_used - old_diffs
            if new_diffs & other_diffs:
                continue
                
            # Move is valid - compute new global max efficiently
            if pi == k:
                cand_row_max = new_val
            else:
                # Since pi < k, the max of the row is current_rows[ri][k]
                # unless new_val > current_rows[ri][k], but that can't happen
                # since new_val < current_rows[ri][pi+1] <= current_rows[ri][k]
                cand_row_max = current_rows[ri][k]
            
            # Update row_maxes temporarily
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max  # restore
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.2:  # Allow some neutral moves to shuffle
                    accept = True
            else:
                # Accept worse moves with probability
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                # Rebuild static diffs for this row
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
        if no_improve > 12000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # 3. Targeted shrink phase: focus on the row(s) with the maximum entry
    polish_deadline = min(deadline, start_time + 113.0)
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # Identify rows that achieve the global max
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    no_improve = 0
    max_no_improve_polish = 20000
    
    # Phase 3a: Try multi-position moves on max rows first
    multi_polish_deadline = min(polish_deadline, start_time + 90.0)
    while time.time() < multi_polish_deadline and no_improve < 1000:
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
    
    # Phase 3b: Single-position polish
    no_improve = 0
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Bias toward rows with the maximum entry
        if max_row_indices and random.random() < 0.7:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        
        # For the max row, prioritize the last position (which is the max)
        if ri in max_row_indices and random.random() < 0.5:
            pi = k  # the last element
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try decreasing moves in polish phase
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        # Try from smallest decrease to largest (more likely valid first)
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
            
            # In polish phase, only accept improving or neutral moves
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.1
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
                    # Update max_row_indices
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
    
    # 4. Final quick SA pass with very low temperature
    final_deadline = deadline
    T = 0.2
    T_min = 0.01
    cool = 0.99999
    
    # Reset state to best
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
        max_decrease = min(20, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
        
        # Try a random subset of decreasing moves
        n_try = min(max_decrease, random.randint(3, 8))
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
                accept = random.random() < 0.05
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
