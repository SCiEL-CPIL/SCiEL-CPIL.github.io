import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, rng=None):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    If rng is provided, use randomized greedy for diversity.
    """
    if rng is None:
        rng = random
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # With some probability, skip a few values to add diversity
                if rng.random() < 0.15 and x < 200:
                    skip = rng.randint(1, 3)
                    x += skip
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        x -= skip  # undo if collision
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

def _random_valid_row(forbidden_diffs, max_limit, k, max_candidates=5, safety_bound=300, rng=None):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_diffs,
    with max entry <= max_limit.
    """
    if rng is None:
        rng = random
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for _ in range(k):
        candidates = []
        x = last_val + 1
        limit = min(max_limit, last_val + safety_bound) if max_limit is not None else last_val + safety_bound
        
        while x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                candidates.append(x)
            
            x += 1
            if len(candidates) >= max_candidates:
                break
        
        if not candidates:
            return None
        
        chosen_x = rng.choice(candidates)
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def _construct_single_start(n_rows, k, rng, deadline):
    """
    Single optimization run: greedy init + row reconstruction + element-level SA.
    Returns (best_rows, best_val).
    """
    rows = _greedy_init(n_rows, k, rng)
    
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    row_maxes = [max(r) for r in rows]
    row_diffs = [_row_diffs_set(r) for r in rows]
    
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    # Phase 1: Row-level reconstruction
    T = 50.0
    T_min = 1.0
    cool = 0.999
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline and no_improve < max_no_improve:
        remaining_time = deadline - time.time()
        if remaining_time < 5.0:
            break
        
        # Preferentially target rows with high max values
        if rng.random() < 0.6:
            max_rows = [i for i in range(n_rows) if row_maxes[i] >= current_val - 5]
            ri = rng.choice(max_rows) if max_rows else rng.randrange(n_rows)
        else:
            ri = rng.randrange(n_rows)
        
        forbidden = set()
        for j in range(n_rows):
            if j != ri:
                forbidden |= row_diffs[j]
        
        # Target: try to beat current global max, or at least not worsen it
        target_max = current_val - 1
        if target_max < k:
            target_max = current_rows[ri][-1] - 1
        
        new_row = _random_valid_row(forbidden, target_max, k, max_candidates=10, rng=rng)
        
        if new_row is None:
            target_max2 = current_val
            new_row = _random_valid_row(forbidden, target_max2, k, max_candidates=10, rng=rng)
        
        if new_row is None:
            target_max3 = current_val + 15
            new_row = _random_valid_row(forbidden, target_max3, k, max_candidates=8, rng=rng)
        
        if new_row is None:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        new_row_max = new_row[-1]
        other_maxes = [row_maxes[j] for j in range(n_rows) if j != ri]
        new_global_max = max(max(other_maxes) if other_maxes else 0, new_row_max)
        
        delta = new_global_max - current_val
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if rng.random() < 0.3:
                accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta / max(T, 0.1))
                if rng.random() < prob:
                    accept = True
        
        if accept:
            old_diffs = row_diffs[ri]
            new_diffs = _row_diffs_set(new_row)
            all_used -= old_diffs
            all_used |= new_diffs
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            row_maxes[ri] = new_row_max
            current_val = new_global_max
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        if no_improve > max_no_improve // 2:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 50.0
            no_improve = 0
    
    # Phase 2: Element-level simulated annealing
    polish_deadline = min(deadline, time.time() + max(0, deadline - time.time()))
    T = 10.0
    T_min = 0.5
    cool = 0.99995
    no_improve = 0
    max_no_improve_polish = 15000
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Bias toward high-value positions in high-max rows
        if rng.random() < 0.5:
            max_rows = [i for i in range(n_rows) if row_maxes[i] >= current_val - 3]
            ri = rng.choice(max_rows) if max_rows else rng.randrange(n_rows)
        else:
            ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        
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
            
        if rng.random() < 0.15:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(rng.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        rng.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            n = len(cand_row)
            new_diffs = set()
            valid_internal = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if d in new_diffs:
                        valid_internal = False
                        break
                    new_diffs.add(d)
                if not valid_internal:
                    break
            
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
                
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
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
            
        T = max(T_min, T * cool)
        
        if no_improve > max_no_improve_polish // 2:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 10.0
            no_improve = 0
    
    return best_rows, best_val

def _final_polish(rows, k, deadline):
    """
    Final polishing: aggressive decreasing moves on the max-value row.
    Returns (best_rows, best_val).
    """
    n_rows = len(rows)
    rng = random
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    T = 1.0
    T_min = 0.05
    cool = 0.99999
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline and no_improve < max_no_improve:
        max_val = current_val
        max_rows = [i for i in range(n_rows) if row_maxes[i] == max_val]
        
        if not max_rows:
            break
            
        ri = rng.choice(max_rows)
        pi = rng.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(20, old_val - min_allowed)
        
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        rng.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            n = len(cand_row)
            new_diffs = set()
            valid_internal = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if d in new_diffs:
                        valid_internal = False
                        break
                    new_diffs.add(d)
                if not valid_internal:
                    break
            
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
                
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.05:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.05))
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
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
            
        T = max(T_min, T * cool)
    
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Multi-start: run several independent optimization passes
    # Allocate time budget across starts
    num_starts = 3
    time_per_start = (deadline - start_time) / num_starts
    
    best_rows = None
    best_val = float('inf')
    
    for s in range(num_starts):
        start_deadline = start_time + (s + 1) * time_per_start
        # Leave some time for final polish
        if s < num_starts - 1:
            start_deadline = min(start_deadline, deadline - 15.0)
        
        # Use a different seed for each start
        seed = s * 1000 + 42
        rng = random.Random(seed)
        
        rows, val = _construct_single_start(n_rows, k, rng, start_deadline)
        
        if val < best_val:
            best_val = val
            best_rows = rows
    
    # Final polish on the best solution
    if best_rows is not None:
        final_deadline = deadline
        polished_rows, polished_val = _final_polish(best_rows, k, final_deadline)
        if polished_val < best_val:
            best_val = polished_val
            best_rows = polished_rows
    
    if best_rows is None:
        # Fallback: pure greedy
        best_rows = _greedy_init(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
