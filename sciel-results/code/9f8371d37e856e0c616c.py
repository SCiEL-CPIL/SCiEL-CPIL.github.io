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
    for row in rows:
        d = _compute_diffs_bitset(row)
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
                # Check new differences
                # new_diffs will be bits for (x - a) for a in row
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

def _local_search(rows, n_rows, k, rng, time_budget):
    """
    Local search focused on reducing the maximum value.
    Uses bitsets for fast conflict checking.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs = [_compute_diffs_bitset(rows[r]) for r in range(n_rows)]
    all_diffs = 0
    for rd in row_diffs:
        all_diffs |= rd
        
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start < time_budget and _time_left() > 2.0:
        if no_improve > max_no_improve:
            break
            
        # Identify the row with the current maximum
        max_vals = [max(r) for r in rows]
        current_max = max(max_vals)
        
        # Pick a row: 70% chance the max row, 30% random
        if rng.random() < 0.7:
            # Pick one of the rows with max value (or close to it)
            candidates = [i for i in range(n_rows) if max_vals[i] >= current_max - 2]
            r = rng.choice(candidates)
        else:
            r = rng.randrange(n_rows)
            
        row = rows[r]
        
        # Pick a position to modify. Bias towards the end (last element is the max).
        # 60% last, 40% random
        if rng.random() < 0.6:
            pos = k
        else:
            pos = rng.randrange(1, k + 1)
            
        old_val = row[pos]
        
        # We want to decrease the value if possible to reduce max.
        # If we can't decrease (already at lower bound), we might need to increase others?
        # For now, focus on decreasing.
        
        lower_bound = row[pos - 1] + 1
        upper_bound = old_val - 1
        
        if upper_bound < lower_bound:
            # Can't decrease. Try to increase? No, that increases max.
            # Maybe we should try to move a *different* element to allow this one to move?
            # For simplicity, skip this move.
            no_improve += 1
            continue
            
        # Generate candidate values to try
        # Try values in descending order (largest reduction first) to find best improvement quickly?
        # Or random sampling if range is large.
        
        if upper_bound - lower_bound < 20:
            candidates_val = list(range(upper_bound, lower_bound - 1, -1))
        else:
            # Sample a few, including the highest possible reduction
            candidates_val = [upper_bound, upper_bound - 1, upper_bound - 5]
            for _ in range(10):
                candidates_val.append(rng.randint(lower_bound, upper_bound))
            # Sort descending to try larger reductions first
            candidates_val = sorted(set(candidates_val), reverse=True)
            
        moved = False
        for new_val in candidates_val:
            if new_val <= row[pos - 1]:
                continue
            if pos < k and new_val >= row[pos + 1]:
                continue
                
            # Check validity
            row[pos] = new_val
            new_rd = _compute_diffs_bitset(row)
            
            # Check internal consistency (distinct diffs within row)
            # Count bits
            if bin(new_rd).count('1') != k * (k + 1) // 2:
                row[pos] = old_val
                continue
                
            # Check conflicts with other rows
            other_diffs = all_diffs & ~row_diffs[r]
            if (new_rd & other_diffs) != 0:
                row[pos] = old_val
                continue
                
            # Valid move
            row_diffs[r] = new_rd
            all_diffs = (all_diffs & ~row_diffs[r]) | new_rd
            
            # Check if max improved
            new_max = max(max(rows[x]) for x in range(n_rows))
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(x) for x in rows]
                no_improve = 0
            else:
                no_improve += 1
                
            # Accept the move (even if max didn't improve, it might help future moves)
            # But for pure descent, we might only accept if it improves max?
            # Let's accept any valid move to explore the space, but track best.
            moved = True
            break
            
        if not moved:
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
    cooling = 0.999
    
    current_max = best_max
    
    while T > T_min and time.time() - start < time_budget and _time_left() > 2.0:
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        row = rows[r]
        old_val = row[pos]
        
        # Random delta
        # Bias towards small changes, but allow larger jumps
        if rng.random() < 0.7:
            delta = rng.randint(-3, 3)
        else:
            delta = rng.randint(-10, 10)
        new_val = old_val + delta
        
        # Bounds
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
        
        # Check internal consistency
        if bin(new_rd).count('1') != k * (k + 1) // 2:
            row[pos] = old_val
            T *= cooling
            continue
            
        # Check conflicts with other rows
        other_diffs = all_diffs & ~row_diffs[r]
        
        if (new_rd & other_diffs) != 0:
            # Conflict exists. Revert.
            row[pos] = old_val
            T *= cooling
            continue
            
        # Valid move
        all_diffs = (all_diffs & ~row_diffs[r]) | new_rd
        row_diffs[r] = new_rd
        
        cur_max = max(max(rows[x]) for x in range(n_rows))
        
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(x) for x in rows]
            
        # Acceptance
        if cur_max <= current_max:
            current_max = cur_max
        else:
            # Uphill move in Max Value
            delta_cost = cur_max - current_max
            if rng.random() < pow(2.71828, -delta_cost / T):
                current_max = cur_max
            else:
                # Revert
                all_diffs = (all_diffs & ~new_rd) | row_diffs[r]
                # row_diffs[r] is still the new one, we need the old one.
                # We didn't store it. Let's recompute or store before.
                # For efficiency, let's store old_rd before modifying.
                # But we already modified row_diffs[r].
                # Let's just recompute old_rd from the restored row? No, row is restored.
                # Wait, row[pos] = old_val is not done yet in the revert block.
                # We need to restore the row and the diff set.
                row[pos] = old_val
                # Recompute old_rd
                old_rd = _compute_diffs_bitset(row)
                row_diffs[r] = old_rd
                all_diffs = (all_diffs & ~new_rd) | old_rd
                # Wait, all_diffs was updated to include new_rd and exclude old_rd.
                # So all_diffs currently has new_rd.
                # We want to remove new_rd and add old_rd.
                # all_diffs = (all_diffs & ~new_rd) | old_rd
                # But we already did all_diffs = (all_diffs & ~row_diffs[r]) | new_rd earlier.
                # So all_diffs has new_rd.
                # We need to remove new_rd and add old_rd.
                # all_diffs = (all_diffs & ~new_rd) | old_rd
                # This is correct.
                
        T *= cooling
        
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Starts
    best_rows = None
    best_max = float('inf')
    
    # Run several greedy builds to find a good starting point
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
        
    # Phase 2: Local Search
    if _time_left() > 10.0:
        budget = min(40.0, _time_left() - 5.0)
        ls_rows, ls_max = _local_search(best_rows, n_rows, k, rng, budget)
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
        # Fallback
        best_rows = _greedy_build(n_rows, k, rng)
        
    return best_rows
