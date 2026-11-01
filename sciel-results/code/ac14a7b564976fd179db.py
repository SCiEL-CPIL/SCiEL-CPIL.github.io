import time
import random
import sys

_START_TIME = None
_DEADLINE = 145.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_bitset(row):
    diffs = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            diffs |= (1 << d)
    return diffs

def _is_valid(rows, n_rows, k):
    seen = 0
    for row in rows:
        d = _compute_diffs_bitset(row)
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, k, rng, order=None):
    if order is None:
        order = list(range(n_rows))
        rng.shuffle(order)
    used = 0
    rows = [None] * n_rows
    for r in order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = 0
                ok = True
                for a in row:
                    d = x - a
                    new_diffs |= (1 << d)
                    if (new_diffs & used) != 0:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    used |= new_diffs
                    break
                x += 1
        rows[r] = row
    return rows

def _rebuild_diffs(rows, n_rows, k):
    row_diffs = [_compute_diffs_bitset(rows[r]) for r in range(n_rows)]
    all_diffs = 0
    for rd in row_diffs:
        all_diffs |= rd
    return row_diffs, all_diffs

def _local_search(rows, n_rows, k, rng, time_budget):
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs, all_diffs = _rebuild_diffs(rows, n_rows, k)
    
    no_improve = 0
    max_no_improve = 800
    
    while time.time() - start < time_budget and _time_left() > 2.0:
        if no_improve > max_no_improve:
            break
            
        max_vals = [max(r) for r in rows]
        current_max = max(max_vals)
        
        if rng.random() < 0.7:
            candidates = [i for i in range(n_rows) if max_vals[i] >= current_max - 3]
            r = rng.choice(candidates)
        else:
            r = rng.randrange(n_rows)
            
        row = rows[r]
        
        if rng.random() < 0.6:
            pos = k
        else:
            pos = rng.randrange(1, k + 1)
            
        old_val = row[pos]
        
        lower_bound = row[pos - 1] + 1
        upper_bound = old_val - 1
        
        if upper_bound < lower_bound:
            no_improve += 1
            continue
            
        if upper_bound - lower_bound < 30:
            candidates_val = list(range(upper_bound, lower_bound - 1, -1))
        else:
            candidates_val = [upper_bound, upper_bound - 1, upper_bound - 3, upper_bound - 7, upper_bound - 15]
            for _ in range(15):
                candidates_val.append(rng.randint(lower_bound, upper_bound))
            candidates_val = sorted(set(candidates_val), reverse=True)
            
        moved = False
        for new_val in candidates_val:
            if new_val <= row[pos - 1]:
                continue
            if pos < k and new_val >= row[pos + 1]:
                continue
                
            row[pos] = new_val
            new_rd = _compute_diffs_bitset(row)
            
            if bin(new_rd).count('1') != k * (k + 1) // 2:
                row[pos] = old_val
                continue
                
            other_diffs = all_diffs & ~row_diffs[r]
            if (new_rd & other_diffs) != 0:
                row[pos] = old_val
                continue
                
            row_diffs[r] = new_rd
            all_diffs = (all_diffs & ~row_diffs[r]) | new_rd
            
            new_max = max(max(rows[x]) for x in range(n_rows))
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(x) for x in rows]
                no_improve = 0
            else:
                no_improve += 1
                
            moved = True
            break
            
        if not moved:
            no_improve += 1
            
    return best_rows, best_max

def _simulated_annealing(rows, n_rows, k, rng, time_budget):
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs, all_diffs = _rebuild_diffs(rows, n_rows, k)
    
    T = 15.0
    T_min = 0.01
    cooling = 0.9995
    
    current_max = best_max
    
    while T > T_min and time.time() - start < time_budget and _time_left() > 2.0:
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        row = rows[r]
        old_val = row[pos]
        old_rd = row_diffs[r]
        
        if rng.random() < 0.7:
            delta = rng.randint(-4, 4)
        else:
            delta = rng.randint(-12, 12)
        new_val = old_val + delta
        
        if new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
            
        row[pos] = new_val
        new_rd = _compute_diffs_bitset(row)
        
        if bin(new_rd).count('1') != k * (k + 1) // 2:
            row[pos] = old_val
            T *= cooling
            continue
            
        other_diffs = all_diffs & ~old_rd
        
        if (new_rd & other_diffs) != 0:
            row[pos] = old_val
            T *= cooling
            continue
            
        all_diffs = (all_diffs & ~old_rd) | new_rd
        row_diffs[r] = new_rd
        
        cur_max = max(max(rows[x]) for x in range(n_rows))
        
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(x) for x in rows]
            
        if cur_max <= current_max:
            current_max = cur_max
        else:
            delta_cost = cur_max - current_max
            if rng.random() < pow(2.71828, -delta_cost / T):
                current_max = cur_max
            else:
                row[pos] = old_val
                row_diffs[r] = old_rd
                all_diffs = (all_diffs & ~new_rd) | old_rd
                
        T *= cooling
        
    return best_rows, best_max

def _multi_restart_sa(rows, n_rows, k, rng, time_budget):
    """
    Run multiple short SA restarts from different perturbations of the initial solution.
    This is a structural change: instead of one long SA, we do many short ones with restarts.
    """
    start = time.time()
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    
    # Each restart gets a portion of the time budget
    num_restarts = 5
    per_restart_budget = (time_budget - 2.0) / num_restarts
    
    for restart in range(num_restarts):
        if _time_left() < 3.0:
            break
            
        # For the first restart, use the original rows. For subsequent ones, perturb best_rows.
        if restart == 0:
            current_rows = [list(r) for r in rows]
        else:
            current_rows = [list(r) for r in best_rows]
            # Perturb: randomly modify a few positions
            for _ in range(rng.randint(1, 3)):
                r = rng.randrange(n_rows)
                pos = rng.randrange(1, k + 1)
                row = current_rows[r]
                old_val = row[pos]
                delta = rng.randint(-5, 5)
                new_val = old_val + delta
                if new_val > row[pos - 1] and (pos == k or new_val < row[pos + 1]) and new_val > 0:
                    # Check if still valid after perturbation
                    row[pos] = new_val
                    rd = _compute_diffs_bitset(row)
                    if bin(rd).count('1') == k * (k + 1) // 2:
                        # Check against other rows
                        row_diffs, all_diffs = _rebuild_diffs(current_rows, n_rows, k)
                        if _is_valid(current_rows, n_rows, k):
                            pass
                        else:
                            row[pos] = old_val
                    else:
                        row[pos] = old_val
                else:
                    row[pos] = old_val
            
        remaining = min(per_restart_budget, _time_left() - 2.0)
        if remaining < 1.0:
            break
            
        sa_rows, sa_max = _simulated_annealing(current_rows, n_rows, k, rng, remaining)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
            
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    best_rows = None
    best_max = float('inf')
    
    # Phase 1: Multiple Greedy Starts
    num_greedy = 30
    for i in range(num_greedy):
        if _time_left() < 8.0:
            break
        rows = _greedy_build(n_rows, k, rng)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, rng)
        best_max = max(max(r) for r in best_rows)
        
    # Phase 2: Local Search
    if _time_left() > 8.0:
        budget = min(35.0, _time_left() - 4.0)
        ls_rows, ls_max = _local_search(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
            
    # Phase 3: Multi-restart Simulated Annealing
    if _time_left() > 8.0:
        budget = min(95.0, _time_left() - 4.0)
        sa_rows, sa_max = _multi_restart_sa(best_rows, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
            
    # Final Verification
    if not _is_valid(best_rows, n_rows, k):
        best_rows = _greedy_build(n_rows, k, rng)
        
    return best_rows
