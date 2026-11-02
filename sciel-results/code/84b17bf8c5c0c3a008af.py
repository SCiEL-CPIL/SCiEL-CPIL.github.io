"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _row_diff_set(row):
    """Compute the set of all positive differences in a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Compute the union of all positive differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diff_set(row)
    return used

def _check_valid_fast(rows, other_diffs):
    """Check if rows are valid given that other_diffs are already used.
    Returns (is_valid, new_diffs) where new_diffs is the set of diffs in rows."""
    new_diffs = set()
    for row in rows:
        rd = _row_diff_set(row)
        if rd & other_diffs:
            return False, None
        new_diffs |= rd
    return True, new_diffs

def _scope(rows):
    """Return the maximum entry (scope) of the rows."""
    return max(max(row) for row in rows)

def _greedy_construct(n_rows, k, rng, strategy='nearest'):
    """Greedy construction with randomization for multi-start.
    strategy: 'nearest' (pick smallest valid), 'random' (pick random from candidates),
              'spread' (pick larger values to spread out)"""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Collect several candidates
            candidates = []
            max_candidates = 5 if strategy == 'nearest' else 15
            for trial in range(max_candidates):
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    candidates.append((x, new))
                    if len(candidates) >= max_candidates:
                        break
            if not candidates:
                # Fallback: keep searching
                while True:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        candidates.append((x, new))
                        break
            
            if strategy == 'nearest':
                x, new = candidates[0]
            elif strategy == 'random':
                x, new = rng.choice(candidates)
            elif strategy == 'spread':
                # Pick from the latter half
                idx = rng.randint(len(candidates) // 2, len(candidates) - 1)
                x, new = candidates[idx]
            else:
                x, new = candidates[0]
                
            row.append(x)
            used |= new
        rows.append(row)
    return rows

def _rebuild_row(row, other_diffs, rng, max_scope=None):
    """Rebuild a single row from scratch given the diffs already used by other rows.
    Returns the new row or None if impossible within max_scope."""
    if max_scope is None:
        max_scope = row[-1] + 10 if row else 100
    new_row = [0]
    x = 0
    k_needed = len(row)  # total elements needed
    
    while len(new_row) < k_needed:
        found = False
        for trial in range(max_scope):
            x += 1
            if x > max_scope:
                return None
            new = {x - a for a in new_row}
            if not new & other_diffs:
                new_row.append(x)
                other_diffs = other_diffs | new
                found = True
                break
        if not found:
            return None
    return new_row

def _local_compact(rows, n_rows, k, rng, time_budget=10.0):
    """Aggressively compact rows by trying to reduce entries via row rebuilding."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    start = time.time()
    
    # Compute diffs for each row
    row_diffs = [_row_diff_set(r) for r in best_rows]
    
    # Iteratively try to compact each row
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        for r_idx in range(n_rows):
            if time.time() - start > time_budget:
                break
            # Compute diffs of all other rows
            other_diffs = set()
            for i in range(n_rows):
                if i != r_idx:
                    other_diffs |= row_diffs[i]
            
            # Try to rebuild this row with a smaller max scope
            current_max = best_rows[r_idx][-1]
            # Try reducing scope progressively
            for new_max in range(current_max - 1, 0, -1):
                if new_max < k:  # minimum possible
                    break
                new_row = _rebuild_row(best_rows[r_idx], other_diffs, rng, max_scope=new_max)
                if new_row is not None:
                    # Verify it's valid
                    new_diff = _row_diff_set(new_row)
                    if not (new_diff & other_diffs):
                        best_rows[r_idx] = new_row
                        row_diffs[r_idx] = new_diff
                        new_scope = _scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        improved = True
                        break
    
    return best_rows, best_scope

def _simulated_annealing(rows, n_rows, k, rng, time_budget=30.0):
    """Simulated annealing to minimize scope. Uses faster validity checks."""
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    start = time.time()
    
    # Precompute row diffs
    current_row_diffs = [_row_diff_set(r) for r in current_rows]
    
    T = 5.0
    cooling = 0.998
    min_T = 0.01
    
    while T > min_T and (time.time() - start) < time_budget:
        # Pick a random row to perturb
        r_idx = rng.randint(0, n_rows - 1)
        row = list(current_rows[r_idx])
        
        # Choose perturbation type
        pert_type = rng.random()
        
        if pert_type < 0.5:
            # Move one element
            pos = rng.randint(1, len(row) - 1)
            delta = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 6, 10])
            new_val = row[pos] + delta
            
            # Ensure increasing order
            if pos > 0 and new_val <= row[pos - 1]:
                new_val = row[pos - 1] + 1
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                new_val = row[pos + 1] - 1
            if new_val <= 0:
                continue
            row[pos] = new_val
        elif pert_type < 0.8:
            # Try to reduce last element
            if len(row) > 1:
                new_last = row[-1] - rng.randint(1, 5)
                if new_last <= row[-2]:
                    continue
                row[-1] = new_last
            else:
                continue
        else:
            # Small random rebuild of last few elements
            n_tail = min(3, len(row) - 1)
            for pos in range(len(row) - n_tail, len(row)):
                if pos == 0:
                    continue
                delta = rng.randint(-3, 3)
                new_val = row[pos] + delta
                if new_val <= row[pos - 1]:
                    new_val = row[pos - 1] + 1
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                row[pos] = new_val
        
        # Check validity: compute new diffs for this row only
        new_row_diff = _row_diff_set(row)
        
        # Check against other rows' diffs
        conflict = False
        for i in range(n_rows):
            if i != r_idx:
                if new_row_diff & current_row_diffs[i]:
                    conflict = True
                    break
        
        if conflict:
            # Reject
            continue
        
        # Accept based on annealing
        new_scope = max(max(row), max(max(r) for i, r in enumerate(current_rows) if i != r_idx))
        delta_scope = new_scope - current_scope
        
        if delta_scope <= 0 or rng.random() < np.exp(-delta_scope / max(T, 0.01)):
            current_rows[r_idx] = row
            current_row_diffs[r_idx] = new_row_diff
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        
        T *= cooling
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Construct difference triangle set with minimal scope."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multi-start greedy constructions with different strategies
    strategies = ['nearest', 'random', 'spread', 'nearest', 'random', 'nearest', 'spread', 'random']
    for i, strat in enumerate(strategies):
        if time.time() > deadline - 60:
            break
        rows = _greedy_construct(n_rows, k, rng, strategy=strat)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback to simple greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Phase 2: Local compaction on best solution
    time_left = deadline - time.time()
    compact_budget = min(20.0, time_left * 0.25)
    if compact_budget > 2:
        best_rows, best_scope = _local_compact(best_rows, n_rows, k, rng, time_budget=compact_budget)
    
    # Phase 3: Simulated annealing with remaining time
    time_left = deadline - time.time()
    sa_budget = max(0, time_left - 5)
    if sa_budget > 3:
        best_rows, best_scope = _simulated_annealing(
            best_rows, n_rows, k, rng, time_budget=sa_budget
        )
    
    # Phase 4: Final compaction pass if time remains
    time_left = deadline - time.time()
    if time_left > 5:
        best_rows, best_scope = _local_compact(best_rows, n_rows, k, rng, time_budget=time_left - 3)
    
    return best_rows

# EVOLVE-BLOCK-END
