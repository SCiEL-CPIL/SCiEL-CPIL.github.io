import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Phase 1: Get initial solution via randomized greedy
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple random seeds for greedy construction
    for seed in range(50):
        if time.time() > deadline - 60:
            break
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: simple greedy without randomization
        best_rows = _greedy_construct(n_rows, k, 0)
        best_scope = _scope(best_rows)
    
    # Phase 2: Local search / simulated annealing to improve
    if time.time() < deadline - 10:
        improved = _local_search(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Try to reduce scope by replacing rows
    if time.time() < deadline - 5:
        further = _row_replacement_search(best_rows, n_rows, k, start_time, deadline)
        if further is not None:
            f_scope = _scope(further)
            if f_scope < best_scope:
                best_scope = f_scope
                best_rows = further
    
    # Phase 4: Final polish - try to reduce each element
    if time.time() < deadline - 2:
        polished = _polish(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            p_scope = _scope(polished)
            if p_scope < best_scope:
                best_scope = p_scope
                best_rows = polished
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid_row(row):
    """Check if a row is strictly increasing with distinct internal differences."""
    n = len(row)
    for i in range(n - 1):
        if row[i + 1] <= row[i]:
            return False
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _greedy_construct(n_rows, k, seed):
    """Construct n_rows rows greedily with randomization."""
    rng = random.Random(seed)
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            # Try to find a valid next element
            found = False
            for x in range(last_x + 1, last_x + 200):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    # Check internal validity
                    candidate = row + [x]
                    if _is_valid_row(candidate):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
            
            if not found:
                # Try harder - search further
                for x in range(last_x + 200, last_x + 500):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidate = row + [x]
                        if _is_valid_row(candidate):
                            row.append(x)
                            used_diffs.update(new_diffs)
                            found = True
                            break
            
            if not found:
                return None  # Failed to construct
        
        rows.append(row)
    
    return rows

def _local_search(initial_rows, n_rows, k, start_time, deadline):
    """Local search with simulated annealing to minimize scope."""
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute diff sets
    row_diff_sets = [_row_diffs(r) for r in rows]
    
    T = 10.0
    cool_rate = 0.999
    
    max_steps = 100000
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
        
        # Pick a random row to modify
        idx = rng.randint(0, n_rows - 1)
        old_row = list(rows[idx])
        old_diffs = row_diff_sets[idx]
        
        # Generate perturbation
        new_row = _perturb_row(old_row, k, rng)
        if new_row is None:
            continue
        
        new_diffs = _row_diffs(new_row)
        
        # Check conflicts with other rows
        conflict = False
        for j in range(n_rows):
            if j == idx:
                continue
            if new_diffs & row_diff_sets[j]:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Check internal validity
        if not _is_valid_row(new_row):
            continue
        
        # Compute new scope
        new_rows = rows[:]
        new_rows[idx] = new_row
        new_scope = _scope(new_rows)
        
        delta = new_scope - current_scope
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            rows = new_rows
            row_diff_sets[idx] = new_diffs
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        
        T *= cool_rate
    
    return best_rows

def _perturb_row(old_row, k, rng):
    """Generate a perturbed version of a row."""
    n = len(old_row)
    candidate = list(old_row)
    
    # Try to modify 1-3 positions
    num_mods = rng.randint(1, 3)
    positions = rng.sample(range(1, n), min(num_mods, n - 1))
    
    for pos in positions:
        old_val = candidate[pos]
        # Try nearby values
        for delta in rng.sample(range(-5, 6), 5):
            new_val = old_val + delta
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= candidate[pos - 1]:
                continue
            if pos < n - 1 and new_val >= candidate[pos + 1]:
                continue
            
            candidate[pos] = new_val
            if _is_valid_row(candidate):
                return candidate
            candidate[pos] = old_val
    
    return None

def _row_replacement_search(initial_rows, n_rows, k, start_time, deadline):
    """Try to replace rows with better alternatives."""
    rng = random.Random(123)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for iteration in range(2000):
        if time.time() > deadline - 5:
            break
        
        idx = rng.randint(0, n_rows - 1)
        old_max = rows[idx][-1]
        
        # Collect diffs from other rows
        used_diffs = set()
        for j in range(n_rows):
            if j != idx:
                used_diffs.update(_row_diffs(rows[j]))
        
        # Try to construct a row with smaller max
        for target_max in range(old_max - 1, k, -1):
            if time.time() > deadline - 5:
                break
            
            new_row = _construct_row_limited(used_diffs, k, target_max, rng)
            if new_row is not None:
                rows[idx] = new_row
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                break
    
    return best_rows

def _construct_row_limited(used_diffs, k, max_val, rng):
    """Construct a row with max <= max_val avoiding used_diffs."""
    row = [0]
    for _ in range(k):
        last_x = row[-1]
        valid_xs = []
        
        for x in range(last_x + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                candidate = row + [x]
                if _is_valid_row(candidate):
                    valid_xs.append(x)
        
        if not valid_xs:
            return None
        
        # Choose randomly, biased toward smaller
        if rng.random() < 0.6:
            chosen = valid_xs[0]
        else:
            chosen = rng.choice(valid_xs[:min(10, len(valid_xs))])
        
        row.append(chosen)
        used_diffs.update({chosen - a for a in row[:-1]})
    
    return row

def _polish(initial_rows, n_rows, k, start_time, deadline):
    """Final polish: try to reduce individual elements."""
    rng = random.Random(777)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for idx in range(n_rows):
        if time.time() > deadline - 2:
            break
        
        # Try to reduce each element from the end
        for pos in range(k, 0, -1):
            if time.time() > deadline - 2:
                break
            
            old_val = rows[idx][pos]
            for new_val in range(old_val - 1, rows[idx][pos - 1], -1):
                candidate = list(rows[idx])
                candidate[pos] = new_val
                
                if not _is_valid_row(candidate):
                    continue
                
                new_diffs = _row_diffs(candidate)
                
                # Check conflicts
                conflict = False
                for j in range(n_rows):
                    if j != idx:
                        if new_diffs & _row_diffs(rows[j]):
                            conflict = True
                            break
                
                if not conflict:
                    rows[idx] = candidate
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    break
    
    return best_rows
