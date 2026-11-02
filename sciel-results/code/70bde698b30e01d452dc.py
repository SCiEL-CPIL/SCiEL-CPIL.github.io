"""Difference triangle set construction: minimize the scope (max entry)."""
import time
import random
from itertools import combinations

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _scope(rows):
    return max(max(row) for row in rows)

def _greedy_construct(n_rows, k, order=None, seed=0):
    """Greedy: build rows one by one, each row greedily picks smallest marks."""
    rng = random.Random(seed)
    if order is None:
        order = list(range(n_rows))
    else:
        rng.shuffle(order)
    
    used = set()
    rows = [None] * n_rows
    
    for idx in order:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows[idx] = row
    return rows

def _greedy_with_backtrack(n_rows, k, max_depth=3, seed=0):
    """Greedy with limited backtracking on the last few rows."""
    # For simplicity, just do greedy; backtracking is complex and time-consuming
    return _greedy_construct(n_rows, k, seed=seed)

def _local_search(rows, k, max_iters=5000, seed=0):
    """Try to reduce the scope by local modifications."""
    rng = random.Random(seed)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    if not _is_valid(best_rows):
        return best_rows, best_scope
    
    current_scope = best_scope
    current_rows = [list(r) for r in best_rows]
    
    for iteration in range(max_iters):
        # Pick a random row and try to reduce its max element
        ri = rng.randint(0, len(current_rows) - 1)
        row = current_rows[ri]
        
        # Try removing the last element and re-greedy with a smaller bound
        # Or try a random perturbation
        # Simple approach: try to shrink the last element
        if len(row) > 1:
            old_last = row[-1]
            # Try values from old_last-1 down to old_last - 5
            found = False
            for new_last in range(old_last - 1, max(0, old_last - 6), -1):
                trial_row = list(row[:-1]) + [new_last]
                # Check strictly increasing
                if any(trial_row[i] >= trial_row[i+1] for i in range(len(trial_row)-1)):
                    continue
                # Check differences
                trial_diffs = _row_diffs(trial_row)
                # Check against all other rows' diffs
                conflict = False
                for j, other in enumerate(current_rows):
                    if j == ri:
                        continue
                    if trial_diffs & _row_diffs(other):
                        conflict = True
                        break
                if not conflict:
                    current_rows[ri] = trial_row
                    new_scope = _scope(current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                        best_scope = current_scope
                    found = True
                    break
            if found:
                continue
        
        # Also try perturbing an interior element
        if len(row) > 2:
            pos = rng.randint(1, len(row) - 2)  # not first or last
            old_val = row[pos]
            for delta in rng.choice([-1, 1]):
                new_val = old_val + delta
                if new_val <= row[pos-1] or new_val >= row[pos+1]:
                    continue
                trial_row = list(row)
                trial_row[pos] = new_val
                trial_diffs = _row_diffs(trial_row)
                conflict = False
                for j, other in enumerate(current_rows):
                    if j == ri:
                        continue
                    if trial_diffs & _row_diffs(other):
                        conflict = True
                        break
                if not conflict:
                    current_rows[ri] = trial_row
                    new_scope = _scope(current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                        best_scope = current_scope
    
    return best_rows, best_scope

def _simulated_annealing(rows, k, max_iters=20000, seed=0, start_temp=10.0):
    """SA to try to find a lower scope."""
    rng = random.Random(seed)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    best = [list(r) for r in current]
    best_scope = current_scope
    
    if not _is_valid(current):
        return best, best_scope
    
    temp = start_temp
    cooling = 0.999
    
    for _ in range(max_iters):
        # Pick a random row and a random position to perturb
        ri = rng.randint(0, len(current) - 1)
        row = current[ri]
        if len(row) <= 1:
            continue
        
        # Decide: change last element or an interior element
        if rng.random() < 0.5:
            pos = len(row) - 1
        else:
            pos = rng.randint(1, len(row) - 2)
        
        old_val = row[pos]
        # Perturb by a small random amount
        delta = rng.randint(-3, 3)
        new_val = old_val + delta
        
        # Bounds check
        if new_val <= 0:
            continue
        if pos > 0 and new_val <= row[pos-1]:
            continue
        if pos < len(row) - 1 and new_val >= row[pos+1]:
            continue
        
        trial = list(row)
        trial[pos] = new_val
        trial_diffs = _row_diffs(trial)
        
        # Check conflicts with other rows
        conflict = False
        for j, other in enumerate(current):
            if j == ri:
                continue
            if trial_diffs & _row_diffs(other):
                conflict = True
                break
        
        if conflict:
            continue
        
        new_scope = _scope([current[i] if i != ri else trial for i in range(len(current))])
        delta_scope = new_scope - current_scope
        
        if delta_scope <= 0 or rng.random() < pow(2.718281828, -delta_scope / max(temp, 0.1)):
            current[ri] = trial
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best = [list(r) for r in current]
        
        temp *= cooling
    
    return best, best_scope

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all pairwise differences distinct across rows."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin for 120s limit
    
    # Phase 1: Get a valid initial solution via greedy
    best_rows = _greedy_construct(n_rows, k, seed=0)
    best_scope = _scope(best_rows)
    
    # Phase 2: Try multiple random orderings for greedy
    for seed in range(1, 20):
        if time.time() > deadline - 30:
            break
        rows = _greedy_construct(n_rows, k, seed=seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 20:
        improved, scope = _local_search(best_rows, k, max_iters=3000, seed=0)
        if scope < best_scope:
            best_rows = improved
            best_scope = scope
    
    # Phase 4: Simulated annealing for further improvement
    if time.time() < deadline - 10:
        sa_result, sa_scope = _simulated_annealing(best_rows, k, max_iters=15000, seed=0)
        if sa_scope < best_scope:
            best_rows = sa_result
            best_scope = sa_scope
    
    # Final validation
    if not _is_valid(best_rows):
        # Fallback to a definitely valid greedy
        best_rows = _greedy_construct(n_rows, k, seed=0)
    
    return best_rows
