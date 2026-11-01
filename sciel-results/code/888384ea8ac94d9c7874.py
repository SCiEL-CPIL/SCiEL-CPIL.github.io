import time
import random
import sys

_START_TIME = None
_DEADLINE = 145.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_bitset(row):
    """Return an integer bitset of all positive differences in a row."""
    diffs = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            diffs |= (1 << d)
    return diffs

def _is_valid(rows, n_rows, k):
    """Check that all differences across all rows are distinct."""
    seen = 0
    expected_total = n_rows * k * (k + 1) // 2
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        d = _compute_diffs_bitset(row)
        if bin(d).count('1') != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, k, rng, order=None):
    """Greedy: build rows in given order using bitsets."""
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

def _try_reduce_last_element(rows, r, used_by_others, k):
    """Try to reduce the last element of row r. Returns (new_row, new_diffs) or (None, None)."""
    row = rows[r]
    old_last = row[k]
    lower = row[k-1] + 1
    if lower >= old_last:
        return None, None
    
    # Try values from old_last - 1 down to lower
    # But only check a subset if range is large
    start_val = old_last - 1
    end_val = lower
    if start_val - end_val > 100:
        # Sample
        candidates = list(range(start_val, end_val - 1, -1))
        if len(candidates) > 50:
            candidates = [candidates[0], candidates[1], candidates[2]] + [random.choice(candidates) for _ in range(47)]
    else:
        candidates = list(range(start_val, end_val - 1, -1))
        
    for new_val in candidates:
        new_row = row[:k] + [new_val]
        new_diffs = _compute_diffs_bitset(new_row)
        if bin(new_diffs).count('1') != k * (k + 1) // 2:
            continue
        if new_diffs & used_by_others:
            continue
        return new_row, new_diffs
    return None, None

def _local_search_aggressive(rows, n_rows, k, rng, time_budget):
    """
    Aggressive local search: repeatedly try to reduce the maximum element across all rows.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs = [_compute_diffs_bitset(rows[r]) for r in range(n_rows)]
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() - start < time_budget and _time_left() > 2.0:
        if no_improve > max_no_improve:
            break
            
        # Compute max for each row
        max_vals = [max(r) for r in rows]
        current_max = max(max_vals)
        
        # Find the row with the maximum value
        # If multiple, pick one randomly
        max_rows = [i for i in range(n_rows) if max_vals[i] == current_max]
        r = rng.choice(max_rows)
        
        # Compute used_by_others
        used_by_others = 0
        for i in range(n_rows):
            if i != r:
                used_by_others |= row_diffs[i]
                
        # Try to reduce the last element of row r
        new_row, new_diffs = _try_reduce_last_element(rows, r, used_by_others, k)
        
        if new_row is not None:
            rows[r] = new_row
            row_diffs[r] = new_diffs
            new_max = max(max(r) for r in rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Try to reduce a random element in the row
            pos = rng.randrange(1, k)
            old_val = rows[r][pos]
            lower = rows[r][pos-1] + 1
            upper = (rows[r][pos+1] - 1) if pos < k else old_val - 1
            if upper >= lower:
                # Try to reduce
                candidates = list(range(upper, lower - 1, -1))
                if len(candidates) > 20:
                    candidates = [candidates[0], candidates[1]] + [random.choice(candidates) for _ in range(18)]
                for new_val in candidates:
                    new_row = rows[r][:]
                    new_row[pos] = new_val
                    new_diffs = _compute_diffs_bitset(new_row)
                    if bin(new_diffs).count('1') != k * (k + 1) // 2:
                        continue
                    if new_diffs & used_by_others:
                        continue
                    rows[r] = new_row
                    row_diffs[r] = new_diffs
                    new_max = max(max(r) for r in rows)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    break
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
    return best_rows, best_max

def _simulated_annealing(rows, n_rows, k, rng, time_budget):
    """
    SA using bitsets for fast conflict checking.
    Objective: Minimize max value.
    Allows uphill moves to escape local minima.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs = [_compute_diffs_bitset(rows[r]) for r in range(n_rows)]
    all_diffs = 0
    for rd in row_diffs:
        all_diffs |= rd
        
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    current_max = best_max
    
    while T > T_min and time.time() - start < time_budget and _time_left() > 2.0:
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        row = rows[r]
        old_val = row[pos]
        
        # Random delta
        if rng.random() < 0.7:
            delta = rng.randint(-3, 3)
        else:
            delta = rng.randint(-10, 10)
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
            
        other_diffs = all_diffs & ~row_diffs[r]
        
        if (new_rd & other_diffs) != 0:
            row[pos] = old_val
            T *= cooling
            continue
            
        old_rd = row_diffs[r]
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

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Starts
    best_rows = None
    best_max = float('inf')
    
    num_greedy = 20
    for i in range(num_greedy):
        if _time_left() < 10.0:
            break
        rows = _greedy_build(n_rows, k, rng)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, rng)
        best_max = max(max(r) for r in best_rows)
        
    # Phase 2: Aggressive Local Search
    if _time_left() > 10.0:
        budget = min(40.0, _time_left() - 5.0)
        ls_rows, ls_max = _local_search_aggressive(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
            
    # Phase 3: Simulated Annealing
    if _time_left() > 10.0:
        budget = min(80.0, _time_left() - 5.0)
        sa_rows, sa_max = _simulated_annealing(best_rows, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
            
    # Final Verification
    if not _is_valid(best_rows, n_rows, k):
        best_rows = _greedy_build(n_rows, k, rng)
        
    return best_rows
