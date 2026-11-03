"""Rewritten program using simulated annealing / local search for difference triangle set."""
import random
import math
import time

def _row_diffs(row):
    """Return set of all positive differences within a single row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _objective(rows):
    """Return the largest entry (scope). Lower is better."""
    return max(max(row) for row in rows)

def _random_row(k, max_val):
    """Generate a random strictly increasing row of k+1 integers starting at 0."""
    if max_val < k:
        max_val = k
    if max_val <= k:
        vals = list(range(1, max_val + 1))
        random.shuffle(vals)
        vals = sorted(vals[:k])
    else:
        vals = sorted(random.sample(range(1, max_val + 1), k))
    return [0] + vals

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all pairwise differences
    within rows are distinct across all rows.
    
    Uses simulated annealing with local search and precomputed diff sets.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * n_rows] if k == 0 else []
    
    start_time = time.time()
    time_budget = 115.0  # Leave 5s buffer
    
    # Total number of differences needed: n_rows * C(k+1, 2)
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Counting bound
    counting_bound = total_diffs
    
    # Phase 1: Greedy initialization
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    best_score = _objective(rows)
    best_rows = [r[:] for r in rows]
    
    # Precompute diff sets for each row
    row_diff_sets = [_row_diffs(r) for r in rows]
    
    # Phase 2: Simulated annealing with precomputed sets
    current_rows = [r[:] for r in rows]
    current_score = best_score
    
    # Temperature schedule - adaptive
    T0 = max(10.0, best_score * 0.1)
    T_min = 0.001
    T = T0
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_budget:
        if T < T_min:
            break
        
        # Pick a random row to modify
        ri = rng.randrange(n_rows)
        old_row = current_rows[ri][:]
        
        new_row = old_row[:]
        move_made = False
        
        # Multiple move strategies
        strategy = rng.random()
        
        if strategy < 0.4:
            # Strategy 1: Reduce the last element
            if old_row[-1] > 1:
                lo = old_row[-2] + 1
                hi = old_row[-1] - 1
                if hi >= lo:
                    new_val = rng.randint(lo, hi)
                    new_row[-1] = new_val
                    move_made = True
                else:
                    new_row[-1] = rng.randint(lo, old_row[-1])
                    move_made = True
        
        elif strategy < 0.6:
            # Strategy 2: Modify a random middle element
            if len(new_row) >= 3:
                idx = rng.randrange(1, len(new_row) - 1)
                lo = new_row[idx - 1] + 1
                hi = new_row[idx + 1] - 1
                if hi >= lo:
                    new_val = rng.randint(lo, hi)
                    if new_val != old_row[idx]:
                        new_row[idx] = new_val
                        move_made = True
        
        elif strategy < 0.8:
            # Strategy 3: Regenerate the row with a smaller max
            old_max = old_row[-1]
            min_max = max(counting_bound // n_rows + k, k)
            if old_max > min_max + 1:
                new_max = rng.randint(min_max, old_max - 1)
                new_row = _random_row(k, new_max)
                if new_row != old_row:
                    move_made = True
                else:
                    new_max2 = rng.randint(min_max, old_max)
                    new_row = _random_row(k, new_max2)
                    if new_row != old_row:
                        move_made = True
        
        elif strategy < 0.9:
            # Strategy 4: Shift a range of elements
            if len(new_row) >= 3:
                i = rng.randrange(1, len(new_row) - 1)
                j = rng.randrange(i + 1, len(new_row))
                shift = rng.randint(-5, 5)
                if shift != 0:
                    for idx in range(i, j + 1):
                        old_val = new_row[idx]
                        new_val = old_val + shift
                        lo = new_row[idx - 1] + 1 if idx > 0 else 1
                        hi = new_row[idx + 1] - 1 if idx < len(new_row) - 1 else current_score
                        new_val = max(lo, min(hi, new_val))
                        if new_val != old_val:
                            new_row[idx] = new_val
                            move_made = True
        
        else:
            # Strategy 5: Completely new random row
            old_max = old_row[-1]
            min_max = max(counting_bound // n_rows, k)
            new_max = rng.randint(min_max, old_max)
            new_row = _random_row(k, new_max)
            if new_row != old_row:
                move_made = True
        
        if not move_made:
            continue
        
        # Validate new row
        if new_row[0] != 0:
            continue
        for i in range(1, len(new_row)):
            if new_row[i] <= new_row[i - 1]:
                continue
        
        # Check internal validity of new row
        new_diffs = _row_diffs(new_row)
        if len(new_diffs) != num_diffs_per_row:
            continue
        
        # Check against all other rows using precomputed sets
        valid = True
        for ri2 in range(n_rows):
            if ri2 == ri:
                continue
            if new_diffs & row_diff_sets[ri2]:
                valid = False
                break
        
        if not valid:
            continue
        
        # Compute new score
        new_row_max = max(new_row)
        other_max = 0
        for ri2 in range(n_rows):
            if ri2 != ri:
                m = max(current_rows[ri2])
                if m > other_max:
                    other_max = m
        new_score = max(other_max, new_row_max)
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            # Accept move
            current_rows[ri] = new_row
            row_diff_sets[ri] = new_diffs
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in current_rows]
        
        # Cool down
        T *= 0.99999
        if T < T_min:
            # Reheat occasionally to escape local minima
            if rng.random() < 0.1:
                T = T0 * 0.5
    
    return best_rows
