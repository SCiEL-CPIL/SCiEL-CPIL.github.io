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

def _greedy_init(n_rows, k, rng=None):
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
                if rng.random() < 0.1 and x < 150:
                    skip = rng.randint(1, 2)
                    x += skip
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        x -= skip
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _random_valid_row(forbidden_diffs, max_limit, k, max_candidates=5, safety_bound=200, rng=None):
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

def _greedy_descent_pass(rows, n_rows, k, all_used, row_diffs, row_maxes, deadline):
    """
    Deterministic pass: for each element, try to reduce it to the smallest valid value.
    Returns (rows, all_used, row_diffs, row_maxes, improved).
    """
    n = k + 1
    improved = False
    
    for r_idx in range(n_rows):
        if time.time() > deadline:
            break
        row = rows[r_idx]
        for pos in range(1, n):
            if time.time() > deadline:
                break
            
            old_val = row[pos]
            lower_bound = row[pos - 1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos + 1] - 1)
            
            if upper_bound < lower_bound:
                continue
            
            # Compute diffs involving this position
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    diffs_involving.add(old_val - row[i])
                else:
                    diffs_involving.add(row[i] - old_val)
            
            forbidden_diffs = all_used - diffs_involving
            
            # Try to find the smallest valid value
            found = False
            for new_val in range(lower_bound, upper_bound + 1):
                new_diffs = set()
                valid = True
                for i in range(n):
                    if i == pos:
                        continue
                    if i < pos:
                        d = new_val - row[i]
                    else:
                        d = row[i] - new_val
                    if d in forbidden_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    # Accept: update state
                    all_used -= diffs_involving
                    all_used |= new_diffs
                    row[pos] = new_val
                    row_diffs[r_idx] = _row_diffs_set(row)
                    row_maxes[r_idx] = max(row)
                    improved = True
                    found = True
                    break
            
            if not found:
                continue
    
    return rows, all_used, row_diffs, row_maxes, improved

def _sa_phase(rows, n_rows, k, all_used, row_diffs, row_maxes, deadline, T_start, max_iter):
    """
    Simulated annealing phase with element-level moves.
    Returns (best_rows, best_val, all_used, row_diffs, row_maxes).
    """
    current_rows = [r[:] for r in rows]
    current_val = max(row_maxes)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    T = T_start
    T_min = 0.1
    cool = 0.9995
    no_improve = 0
    
    rng = random.Random(int(time.time() * 1000) % 100000)
    
    while time.time() < deadline and no_improve < max_iter:
        # Bias toward high-value positions
        if rng.random() < 0.5:
            max_rows = [i for i in range(n_rows) if row_maxes[i] >= current_val - 3]
            ri = rng.choice(max_rows) if max_rows else rng.randrange(n_rows)
        else:
            ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
        max_increase = min(10, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        if rng.random() < 0.1:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 15
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
        
        if no_improve > max_iter // 3:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = T_start
            no_improve = 0
    
    return best_rows, best_val, all_used, row_diffs, row_maxes

def _construct_single_start(n_rows, k, rng, deadline):
    """
    Single optimization run: greedy init + row reconstruction + greedy descent + SA.
    """
    rows = _greedy_init(n_rows, k, rng)
    
    all_used = _compute_all_used(rows)
    row_diffs = [_row_diffs_set(r) for r in rows]
    row_maxes = [max(r) for r in rows]
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    # Phase 1: Row-level reconstruction with SA
    T = 50.0
    T_min = 1.0
    cool = 0.999
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 10.0 and no_improve < max_no_improve:
        if rng.random() < 0.6:
            max_rows = [i for i in range(n_rows) if row_maxes[i] >= current_val - 5]
            ri = rng.choice(max_rows) if max_rows else rng.randrange(n_rows)
        else:
            ri = rng.randrange(n_rows)
        
        forbidden = set()
        for j in range(n_rows):
            if j != ri:
                forbidden |= row_diffs[j]
        
        target_max = current_val - 1
        if target_max < k:
            target_max = current_rows[ri][-1] - 1
        
        new_row = _random_valid_row(forbidden, target_max, k, max_candidates=10, rng=rng)
        
        if new_row is None:
            target_max2 = current_val
            new_row = _random_valid_row(forbidden, target_max2, k, max_candidates=10, rng=rng)
        
        if new_row is None:
            target_max3 = current_val + 10
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
    
    # Phase 2: Iterated greedy descent + SA
    # Alternate between deterministic greedy descent and short SA bursts
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    polish_deadline = deadline - 5.0
    sa_budget_per_iter = 2.0
    max_iters = 20
    
    for it in range(max_iters):
        if time.time() > polish_deadline:
            break
        
        # Greedy descent pass
        gd_deadline = min(polish_deadline, time.time() + 5.0)
        current_rows, all_used, row_diffs, row_maxes, improved = _greedy_descent_pass(
            current_rows, n_rows, k, all_used, row_diffs, row_maxes, gd_deadline
        )
        
        new_val = max(row_maxes)
        if new_val < best_val:
            best_val = new_val
            best_rows = [r[:] for r in current_rows]
        
        # If no improvement from greedy descent, try a short SA burst
        if not improved and time.time() < polish_deadline:
            sa_deadline = min(polish_deadline, time.time() + sa_budget_per_iter)
            current_rows, current_val, all_used, row_diffs, row_maxes = _sa_phase(
                current_rows, n_rows, k, all_used, row_diffs, row_maxes,
                sa_deadline, T_start=5.0, max_iter=500
            )
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
        
        # If we've hit a hard local minimum, do a more aggressive SA
        if not improved and time.time() < polish_deadline:
            sa_deadline = min(polish_deadline, time.time() + 3.0)
            current_rows, current_val, all_used, row_diffs, row_maxes = _sa_phase(
                current_rows, n_rows, k, all_used, row_diffs, row_maxes,
                sa_deadline, T_start=15.0, max_iter=1000
            )
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
    
    # Final greedy descent
    if time.time() < deadline:
        current_rows = [r[:] for r in best_rows]
        all_used = _compute_all_used(current_rows)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        current_rows, all_used, row_diffs, row_maxes, _ = _greedy_descent_pass(
            current_rows, n_rows, k, all_used, row_diffs, row_maxes, deadline
        )
        
        final_val = max(row_maxes)
        if final_val < best_val:
            best_val = final_val
            best_rows = current_rows
    
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    num_starts = 3
    time_per_start = (deadline - start_time - 10.0) / num_starts
    
    best_rows = None
    best_val = float('inf')
    
    for s in range(num_starts):
        start_deadline = start_time + (s + 1) * time_per_start
        if s < num_starts - 1:
            start_deadline = min(start_deadline, deadline - 15.0)
        
        seed = s * 1000 + 42
        rng = random.Random(seed)
        
        rows, val = _construct_single_start(n_rows, k, rng, start_deadline)
        
        if val < best_val:
            best_val = val
            best_rows = rows
    
    if best_rows is None:
        best_rows = _greedy_init(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
