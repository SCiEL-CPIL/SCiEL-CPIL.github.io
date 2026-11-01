"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_set(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate_solution(rows, n_rows, k):
    """Check that all differences across all rows are distinct and rows are valid."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k):
            if row[i] >= row[i+1]:
                return False
        d = _compute_diffs_set(row)
        if len(d) != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, k, row_order, rng, randomized=False, rand_prob=0.1, window=30):
    """Greedy: build rows in given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            if randomized:
                valid_candidates = []
                for candidate in range(x, x + window):
                    new_diffs = {candidate - a for a in row}
                    if not (new_diffs & used):
                        valid_candidates.append(candidate)
                        if len(valid_candidates) >= 5:
                            break
                if valid_candidates:
                    if rng.random() < rand_prob and len(valid_candidates) > 1:
                        x = rng.choice(valid_candidates)
                    else:
                        x = valid_candidates[0]
                else:
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            break
                        x += 1
            else:
                while True:
                    ok = True
                    for a in row:
                        d = x - a
                        if d in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
            
            row.append(x)
            for a in row[:-1]:
                used.add(x - a)
        rows[r] = row
    return rows

def _greedy_interleaved(n_rows, k, rng, strategy="smallest"):
    """Interleaved greedy: at each step, pick a row to extend."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    
    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        
        candidates = [r for r in range(n_rows) if len(rows[r]) < k + 1]
        if not candidates:
            break
        
        if strategy == "smallest":
            best_r = min(candidates, key=lambda r: rows[r][-1])
        elif strategy == "largest":
            best_r = max(candidates, key=lambda r: rows[r][-1])
        elif strategy == "random":
            best_r = rng.choice(candidates)
        elif strategy == "balanced":
            best_r = max(candidates, key=lambda r: len(rows[r]))
        else:
            best_r = min(candidates, key=lambda r: rows[r][-1])
        
        row = rows[best_r]
        x = row[-1] + 1
        while True:
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
                placed += 1
                break
            x += 1
    return rows

def _try_greedy_variants(n_rows, k, rng, num_trials=40):
    """Try multiple greedy strategies and return best."""
    best_rows = None
    best_max = float('inf')
    
    strategies = ["smallest", "largest", "random", "balanced"]
    
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rand = (t % 2 == 0)
        rows = _greedy_build(n_rows, k, order, rng, randomized=rand)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        strategy = strategies[t % len(strategies)]
        rows = _greedy_interleaved(n_rows, k, rng, strategy=strategy)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_max

def _simulated_annealing(rows, n_rows, k, rng, time_budget=60.0):
    """
    Simulated Annealing: allow uphill moves to escape local minima.
    Moves: pick a random row and position, try to change its value.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    # Precompute difference sets for each row
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    
    def get_other_diffs(exclude_r):
        """Return set of all differences except those in row exclude_r."""
        all_d = set()
        for r in range(n_rows):
            if r != exclude_r:
                all_d |= row_diffs[r]
        return all_d
    
    # Initial temperature
    T = 50.0
    cooling_rate = 0.999
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        # Pick a random row and position
        r = rng.randrange(n_rows)
        row = rows[r]
        if len(row) <= 1:
            continue
        
        # Pick a random position (not the first)
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        
        # Determine valid range for new value
        lower_bound = row[pos - 1] + 1
        upper_bound = old_val + 10  # Allow both increase and decrease
        if pos < len(row) - 1:
            upper_bound = min(upper_bound, row[pos + 1] - 1)
        
        if lower_bound >= upper_bound:
            continue
        
        # Try a random new value in the range
        new_val = rng.randint(lower_bound, upper_bound)
        if new_val == old_val:
            continue
        
        # Compute new differences for this row
        new_row = row[:]
        new_row[pos] = new_val
        
        # Check if the new row is strictly increasing
        valid_row = True
        for i in range(len(new_row) - 1):
            if new_row[i] >= new_row[i+1]:
                valid_row = False
                break
        if not valid_row:
            continue
        
        new_row_diffs = _compute_diffs_set(new_row)
        
        # Check internal consistency
        if len(new_row_diffs) != len(new_row) * (len(new_row) - 1) // 2:
            continue
        
        # Get other diffs
        other_diffs = get_other_diffs(r)
        
        # Check conflicts
        conflicts = len(new_row_diffs & other_diffs)
        
        # Compute objective change
        old_max = max(max(rows[r]) for r in range(n_rows))
        new_max = max(max(new_row), max(max(rows[r]) for r in range(n_rows) if r != pos))
        
        delta_obj = new_max - old_max
        
        # Accept if improvement or with probability based on temperature
        if delta_obj <= 0 or rng.random() < pow(2.718281828, -delta_obj / T):
            # Accept the move
            row[pos] = new_val
            row_diffs[r] = new_row_diffs
            
            cur_max = max(max(rows[r]) for r in range(n_rows))
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in rows]
        
        # Cool down
        T *= cooling_rate
    
    return best_rows, best_max

def _conflict_directed_repair(rows, n_rows, k, rng, time_budget=60.0):
    """
    Structural optimization: Target the global maximum and try to reduce it.
    Uses conflict-directed repair: replace the largest entry with the smallest integer
    that restores the distinct-difference property.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    # Precompute difference sets for each row
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    
    def get_other_diffs(exclude_r):
        """Return set of all differences except those in row exclude_r."""
        all_d = set()
        for r in range(n_rows):
            if r != exclude_r:
                all_d |= row_diffs[r]
        return all_d

    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find the row with the current global maximum
        max_val = max(max(rows[r]) for r in range(n_rows))
        if max_val < best_max:
            best_max = max_val
            best_rows = [list(r) for r in rows]
        
        # Identify rows containing this max value
        max_rows = [r for r in range(n_rows) if max(rows[r]) == max_val]
        
        improved = False
        for r in max_rows:
            if _time_left() < 3.0:
                break
            row = rows[r]
            last_idx = len(row) - 1
            old_val = row[last_idx]
            
            # Conflict-directed repair: find the smallest valid value for the last element
            lower_bound = row[last_idx - 1] + 1
            upper_bound = old_val - 1
            
            if lower_bound > upper_bound:
                continue
            
            base_diffs = _compute_diffs_set(row[:-1])
            other_diffs = get_other_diffs(r)
            
            found = False
            for new_val in range(lower_bound, upper_bound + 1):
                valid_internal = True
                new_diffs_set = set()
                for a in row[:-1]:
                    d = new_val - a
                    if d in base_diffs or d in new_diffs_set:
                        valid_internal = False
                        break
                    new_diffs_set.add(d)
                
                if not valid_internal:
                    continue
                
                valid_external = True
                for d in new_diffs_set:
                    if d in other_diffs:
                        valid_external = False
                        break
                
                if not valid_external:
                    continue
                
                row[last_idx] = new_val
                row_diffs[r] = _compute_diffs_set(row)
                improved = True
                found = True
                break
            
            if improved:
                break
        
        if not improved:
            # Try to reduce the max of a different row to open up space
            non_max_rows = [r for r in range(n_rows) if max(rows[r]) < max_val]
            if non_max_rows:
                r = rng.choice(non_max_rows)
                row = rows[r]
                last_idx = len(row) - 1
                old_val = row[last_idx]
                lower_bound = row[last_idx - 1] + 1
                upper_bound = old_val - 1
                
                if lower_bound <= upper_bound:
                    base_diffs = _compute_diffs_set(row[:-1])
                    other_diffs = get_other_diffs(r)
                    
                    for new_val in range(lower_bound, upper_bound + 1):
                        valid_internal = True
                        new_diffs_set = set()
                        for a in row[:-1]:
                            d = new_val - a
                            if d in base_diffs or d in new_diffs_set:
                                valid_internal = False
                                break
                            new_diffs_set.add(d)
                        
                        if not valid_internal:
                            continue
                        
                        valid_external = True
                        for d in new_diffs_set:
                            if d in other_diffs:
                                valid_external = False
                                break
                        
                        if not valid_external:
                            continue
                        
                        row[last_idx] = new_val
                        row_diffs[r] = _compute_diffs_set(row)
                        improved = True
                        break
            
            if not improved:
                # Try a random perturbation: pick a random row and position, try to decrease
                r = rng.randrange(n_rows)
                row = rows[r]
                if len(row) > 1:
                    pos = rng.randrange(1, len(row))
                    old_val = row[pos]
                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1
                    if pos < len(row) - 1:
                        upper_bound = min(upper_bound, row[pos + 1] - 1)
                    
                    if lower_bound <= upper_bound:
                        old_diffs = set()
                        for i in range(len(row)):
                            if i != pos:
                                if row[pos] > row[i]:
                                    old_diffs.add(row[pos] - row[i])
                                else:
                                    old_diffs.add(row[i] - row[pos])
                        
                        base_diffs = _compute_diffs_set([row[i] for i in range(len(row)) if i != pos])
                        other_diffs = get_other_diffs(r)
                        
                        for new_val in range(lower_bound, upper_bound + 1):
                            new_diffs = set()
                            for i in range(len(row)):
                                if i != pos:
                                    if new_val > row[i]:
                                        new_diffs.add(new_val - row[i])
                                    else:
                                        new_diffs.add(row[i] - new_val)
                            
                            if new_diffs & base_diffs:
                                continue
                            if new_diffs & other_diffs:
                                continue
                            
                            row[pos] = new_val
                            row_diffs[r] = _compute_diffs_set(row)
                            improved = True
                            break
                
                if not improved:
                    break

    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Fast greedy to get a baseline
    best_rows, best_max = _try_greedy_variants(n_rows, k, rng, num_trials=50)
    
    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
        
    # Phase 2: Conflict-Directed Repair
    if _time_left() > 15.0:
        ls_rows, ls_max = _conflict_directed_repair(best_rows, n_rows, k, rng, time_budget=_time_left() - 10.0)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Phase 3: Simulated Annealing to escape local minima
    if _time_left() > 20.0:
        sa_rows, sa_max = _simulated_annealing(best_rows, n_rows, k, rng, time_budget=min(30.0, _time_left() - 10.0))
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Phase 4: If time remains, try another round with a different seed
    if _time_left() > 20.0:
        rng2 = random.Random(123)
        alt_rows, alt_max = _try_greedy_variants(n_rows, k, rng2, num_trials=20)
        if alt_rows is not None:
            alt_refined_rows, alt_refined_max = _conflict_directed_repair(alt_rows, n_rows, k, rng2, time_budget=min(15.0, _time_left() - 10.0))
            if alt_refined_max < best_max:
                best_max = alt_refined_max
                best_rows = alt_refined_rows
            
            # Also try SA on the alternative
            if _time_left() > 15.0:
                alt_sa_rows, alt_sa_max = _simulated_annealing(alt_refined_rows, n_rows, k, rng2, time_budget=min(10.0, _time_left() - 5.0))
                if alt_sa_max < best_max:
                    best_max = alt_sa_max
                    best_rows = alt_sa_rows

    # Final verification
    if not _validate_solution(best_rows, n_rows, k):
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
        
    return best_rows
