"""Difference Triangle Set (4,7) solver using simulated annealing with local search."""
import time
import random
import math

def _row_diffs(row):
    """Compute all positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs

def _check_valid(rows):
    """Check if all positive differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        d = _row_diffs(row)
        if d & all_diffs:
            return False
        all_diffs |= d
    return True

def _max_entry(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k):
    """Deterministic greedy: build each row by scanning upward, picking smallest valid value."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
            x += 1
        rows.append(row)
    return rows

def _greedy_randomized(n_rows, k, rng, max_tries=2000):
    """Randomized greedy: build each row with random candidate selection."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        local = set()
        tries = 0
        while len(row) < k + 1 and tries < max_tries:
            tries += 1
            # Random value in a reasonable range
            x = rng.randint(row[-1] + 1, 200)
            new = {x - a for a in row}
            if not (new & used) and not (new & local):
                row.append(x)
                local |= new
        if len(row) == k + 1:
            rows.append(row)
            used |= local
        else:
            return None
    return rows

def _try_replace_row(rows, row_idx, k, used_others, rng, max_tries=500):
    """Try to find a valid replacement for rows[row_idx] that avoids used_others."""
    used = set(used_others)
    row = [0]
    local = set()
    tries = 0
    while len(row) < k + 1 and tries < max_tries:
        tries += 1
        # Bias toward values that spread out
        lo = row[-1] + 1
        hi = 200
        if lo > hi:
            return None
        x = rng.randint(lo, hi)
        new = {x - a for a in row}
        if not (new & used) and not (new & local):
            row.append(x)
            local |= new
    if len(row) == k + 1:
        return row, local
    return None

def _perturb_row(row, k, used_others, rng, num_perturbations=3):
    """Perturb a row by moving some elements, maintaining validity."""
    row = list(row)
    for _ in range(num_perturbations):
        # Pick a random index to move (not 0)
        idx = rng.randint(1, len(row) - 1)
        old_val = row[idx]
        # Try several new values
        for _ in range(50):
            delta = rng.randint(-10, 10)
            new_val = old_val + delta
            if new_val <= row[idx - 1] or (idx < len(row) - 1 and new_val >= row[idx + 1]):
                continue
            # Check if moving this element maintains validity
            # Remove old differences involving this element
            new_row = row[:]
            new_row[idx] = new_val
            # Check internal validity
            internal_ok = True
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d <= 0:
                        internal_ok = False
                        break
                if not internal_ok:
                    break
            if not internal_ok:
                continue
            # Check against used_others
            d = _row_diffs(new_row)
            if d & used_others:
                continue
            row = new_row
            break
    return row

def _simulated_annealing(n_rows, k, initial_rows, time_limit, rng):
    """Simulated annealing to minimize max entry."""
    rows = [list(r) for r in initial_rows]
    current_scope = _max_entry(rows)
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    
    # Precompute used diffs for each row
    row_diffs = [_row_diffs(r) for r in rows]
    all_diffs = set()
    for d in row_diffs:
        all_diffs |= d
    
    t_start = time.time()
    t = 100.0  # Initial temperature
    cooling_rate = 0.999
    
    while time.time() - t_start < time_limit and t > 0.1:
        t *= cooling_rate
        
        # Choose a random row to modify
        row_idx = rng.randint(0, n_rows - 1)
        
        # Compute used diffs excluding this row
        used_others = set(all_diffs)
        used_others.difference_update(row_diffs[row_idx])
        
        # Try to find a new row
        new_row, new_diffs = _try_replace_row(rows, row_idx, k, used_others, rng, max_tries=200)
        
        if new_row is None:
            continue
        
        new_scope = max(max(r) for r in rows[:row_idx] + [new_row] + rows[row_idx+1:])
        
        if new_scope < current_scope or (new_scope == current_scope and rng.random() < 0.3):
            # Accept
            rows[row_idx] = new_row
            row_diffs[row_idx] = new_diffs
            all_diffs = set()
            for d in row_diffs:
                all_diffs |= d
            current_scope = new_scope
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        elif rng.random() < math.exp(-(new_scope - current_scope) / t):
            # Accept worse solution probabilistically
            rows[row_idx] = new_row
            row_diffs[row_idx] = new_diffs
            all_diffs = set()
            for d in row_diffs:
                all_diffs |= d
            current_scope = new_scope
    
    return best_rows, best_scope

def _local_search(n_rows, k, rows, time_limit, rng):
    """Local search: repeatedly try to improve individual rows."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(rows)
    
    t_start = time.time()
    no_improve = 0
    max_no_improve = 100
    
    while time.time() - t_start < time_limit and no_improve < max_no_improve:
        improved = False
        for row_idx in range(n_rows):
            # Compute used diffs excluding this row
            used_others = set()
            for j in range(n_rows):
                if j != row_idx:
                    used_others |= _row_diffs(rows[j])
            
            # Try to find a better row
            current_row = rows[row_idx]
            current_max = max(current_row)
            
            best_new_row = None
            best_new_max = current_max
            
            for _ in range(100):
                new_row, new_diffs = _try_replace_row(rows, row_idx, k, used_others, rng, max_tries=100)
                if new_row is not None:
                    new_max = max(new_row)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_new_row = new_row
            
            if best_new_row is not None and best_new_max < current_max:
                rows[row_idx] = best_new_row
                improved = True
        
        if improved:
            no_improve = 0
            current_scope = _max_entry(rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            no_improve += 1
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Main entry point: construct n_rows rows of k+1 increasing integers starting at 0."""
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin under 150s
    rng = random.Random(42)
    
    # Phase 1: Generate multiple initial solutions
    initial_solutions = []
    
    # Deterministic greedy
    sol1 = _greedy_construct(n_rows, k)
    initial_solutions.append(sol1)
    
    # Randomized greedy with different seeds
    for seed in range(10):
        rng2 = random.Random(seed * 1000 + 7)
        sol = _greedy_randomized(n_rows, k, rng2, max_tries=3000)
        if sol is not None:
            initial_solutions.append(sol)
        if time.time() > deadline - 60:
            break
    
    # Select the best initial solution
    best_initial = min(initial_solutions, key=_max_entry)
    best_scope = _max_entry(best_initial)
    best_rows = [list(r) for r in best_initial]
    
    # Phase 2: Simulated annealing on the best initial solution
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        sa_rows, sa_scope = _simulated_annealing(n_rows, k, best_rows, min(time_remaining - 5, 60), rng)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
    
    # Phase 3: Local search to polish
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        ls_rows, ls_scope = _local_search(n_rows, k, best_rows, min(time_remaining - 2, 30), rng)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows
    
    # Final validity check
    if not _check_valid(best_rows):
        # Fallback to deterministic greedy
        best_rows = _greedy_construct(n_rows, k)
    
    return best_rows
