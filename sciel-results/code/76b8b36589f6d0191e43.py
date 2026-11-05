import time
import random
import math
import sys
import numpy as np


def _row_diffs_list(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_mask(row):
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    seen = set()
    for row in rows:
        for d in _row_diffs_list(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _compute_all_diffs(rows):
    """Return list of all pairwise differences across all rows."""
    diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
    return diffs


def _greedy_construct(n_rows, k, max_scope=200, seed=42):
    """
    Greedy construction: for each row, pick elements one by one,
    choosing the smallest value that doesn't conflict with any used difference.
    """
    rng = random.Random(seed)
    used_diffs = set()
    rows = []
    for r in range(n_rows):
        row = [0]
        for pos in range(1, k + 1):
            found = False
            for val in range(row[-1] + 1, max_scope + 1):
                new_diffs = set(val - prev for prev in row)
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                if new_diffs & used_diffs:
                    continue
                row.append(val)
                used_diffs |= new_diffs
                found = True
                break
            if not found:
                # Fallback: pick a random larger value
                val = row[-1] + rng.randint(10, 30)
                row.append(val)
                for d in (val - prev for prev in row[:-1]):
                    used_diffs.add(d)
        rows.append(row)
    return rows


def _backtrack_construct_row(row_prefix, used_diffs, k, max_scope, seed=42):
    """
    Backtrack to construct a single row given a prefix and the set of already-used differences.
    Returns the completed row or None if no solution found within max_scope.
    """
    rng = random.Random(seed)
    row = list(row_prefix)
    
    def backtrack():
        if len(row) == k + 1:
            return True
        last = row[-1]
        # Try values in increasing order, with some randomization
        candidates = list(range(last + 1, max_scope + 1))
        rng.shuffle(candidates)
        # Sort by a heuristic: prefer values that create "rare" differences
        # (differences not yet used and that leave more room for future elements)
        def score(val):
            new_diffs = set(val - prev for prev in row)
            if new_diffs & used_diffs:
                return float('inf')
            # Score: prefer smaller values (tighter scope) and differences that are "spread out"
            return val * 1.0 + sum(d * 0.01 for d in new_diffs)
        candidates.sort(key=score)
        for val in candidates:
            new_diffs = set(val - prev for prev in row)
            if new_diffs & used_diffs:
                continue
            if len(new_diffs) != len(set(new_diffs)):
                continue
            row.append(val)
            used_diffs |= new_diffs
            if backtrack():
                return True
            used_diffs -= new_diffs
            row.pop()
        return False
    
    if backtrack():
        return row
    return None


def _construct_with_backtracking(n_rows, k, max_scope=200, seed=42):
    """
    Construct rows one by one using backtracking for each row.
    """
    rng = random.Random(seed)
    used_diffs = set()
    rows = []
    for r in range(n_rows):
        row_prefix = [0]
        row = _backtrack_construct_row(row_prefix, used_diffs, k, max_scope, seed=seed + r * 1000)
        if row is None:
            # Fallback: greedy
            row = [0]
            for pos in range(1, k + 1):
                found = False
                for val in range(row[-1] + 1, max_scope + 1):
                    new_diffs = set(val - prev for prev in row)
                    if len(new_diffs) != len(set(new_diffs)):
                        continue
                    if new_diffs & used_diffs:
                        continue
                    row.append(val)
                    used_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    val = row[-1] + rng.randint(10, 30)
                    row.append(val)
                    for d in (val - prev for prev in row[:-1]):
                        used_diffs.add(d)
        rows.append(row)
        for d in _row_diffs_list(row):
            used_diffs.add(d)
    return rows


def _try_move_element(rows, row_idx, pos, new_val):
    """
    Try to move element at (row_idx, pos) to new_val.
    Returns True if the move is valid (no new conflicts), False otherwise.
    """
    row = rows[row_idx]
    old_val = row[pos]
    if new_val == old_val:
        return True
    # Check bounds
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < len(row) - 1 and new_val >= row[pos + 1]:
        return False
    
    # Compute new differences for this row
    row[pos] = new_val
    new_diffs = set(_row_diffs_list(row))
    row[pos] = old_val
    
    # Check for conflicts with other rows
    all_other_diffs = set()
    for i, r in enumerate(rows):
        if i == row_idx:
            continue
        all_other_diffs |= set(_row_diffs_list(r))
    
    if new_diffs & all_other_diffs:
        return False
    
    # Also check internal uniqueness (already guaranteed by strict increase)
    return True


def _simulated_annealing(rows, k, deadline, seed=42, initial_temp=50.0):
    """
    Simulated annealing to reduce the scope while maintaining validity.
    """
    rng = random.Random(seed)
    cur_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    temp = initial_temp
    cooling = 0.9999
    min_temp = 0.01
    
    def get_conflict_count():
        diffs = _compute_all_diffs(cur_rows)
        return len(diffs) - len(set(diffs))
    
    cur_conf = get_conflict_count()
    cur_scope = _scope(cur_rows)
    cur_cost = cur_conf * 10 + cur_scope
    
    while time.time() < deadline and temp > min_temp:
        # Pick a random element to move
        row_idx = rng.randint(0, len(cur_rows) - 1)
        pos = rng.randint(1, len(cur_rows[row_idx]) - 1)
        old_val = cur_rows[row_idx][pos]
        
        # Propose a new value
        delta = rng.randint(-5, 5)
        if delta == 0:
            temp *= cooling
            continue
        new_val = old_val + delta
        
        # Check bounds
        low = cur_rows[row_idx][pos - 1] + 1
        high = cur_rows[row_idx][pos + 1] - 1 if pos < len(cur_rows[row_idx]) - 1 else 300
        if new_val < low or new_val > high:
            temp *= cooling
            continue
        
        # Apply move
        cur_rows[row_idx][pos] = new_val
        new_conf = get_conflict_count()
        new_scope = _scope(cur_rows)
        new_cost = new_conf * 10 + new_scope
        
        delta_cost = new_cost - cur_cost
        if delta_cost <= 0 or rng.random() < math.exp(-delta_cost / temp):
            cur_conf = new_conf
            cur_scope = new_scope
            cur_cost = new_cost
            if new_conf == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in cur_rows]
        else:
            cur_rows[row_idx][pos] = old_val
        
        temp *= cooling
    
    return best_rows, best_scope


def _local_search_refine(rows, k, deadline, seed=42):
    """
    Local search: try to reduce scope by moving elements downward,
    only accepting moves that don't create conflicts.
    """
    rng = random.Random(seed)
    cur_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Precompute all differences for fast conflict checking
    def get_diff_set():
        s = set()
        for row in cur_rows:
            s |= set(_row_diffs_list(row))
        return s
    
    cur_diff_set = get_diff_set()
    cur_scope = _scope(cur_rows)
    stagnation = 0
    max_stag = 100000
    
    while time.time() < deadline and stagnation < max_stag:
        row_idx = rng.randint(0, len(cur_rows) - 1)
        pos = rng.randint(1, len(cur_rows[row_idx]) - 1)
        old_val = cur_rows[row_idx][pos]
        
        # Try to move downward
        low = cur_rows[row_idx][pos - 1] + 1
        if low >= old_val:
            stagnation += 1
            continue
        
        new_val = rng.randint(low, old_val - 1)
        if new_val == old_val:
            stagnation += 1
            continue
        
        # Check if move is valid
        row = cur_rows[row_idx]
        row[pos] = new_val
        new_row_diffs = set(_row_diffs_list(row))
        row[pos] = old_val
        
        # Check for conflicts
        other_diffs = set()
        for i, r in enumerate(cur_rows):
            if i == row_idx:
                continue
            other_diffs |= set(_row_diffs_list(r))
        
        if new_row_diffs & other_diffs:
            stagnation += 1
            continue
        
        # Apply move
        cur_rows[row_idx][pos] = new_val
        new_scope = _scope(cur_rows)
        
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [list(r) for r in cur_rows]
            stagnation = 0
        else:
            cur_rows[row_idx][pos] = old_val
            stagnation += 1
    
    return best_rows, best_scope


def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple greedy constructions with different seeds
    for seed in range(5):
        if time.time() > DEADLINE - 100:
            break
        rows = _greedy_construct(n_rows, k, max_scope=200, seed=seed * 7 + 3)
        if _is_valid(rows):
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # Phase 2: Backtracking construction with different max_scope values
    if time.time() < DEADLINE - 80:
        for max_scope in [150, 180, 200]:
            if time.time() > DEADLINE - 80:
                break
            for seed in range(3):
                if time.time() > DEADLINE - 80:
                    break
                rows = _construct_with_backtracking(n_rows, k, max_scope=max_scope, seed=seed * 13 + 11)
                if _is_valid(rows):
                    scope = _scope(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = rows
    
    # Fallback if no valid solution found
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)
    
    # Phase 3: Simulated annealing to reduce scope
    if time.time() < DEADLINE - 40:
        for seed in range(3):
            if time.time() > DEADLINE - 35:
                break
            sa_rows, sa_scope = _simulated_annealing(
                best_rows, k, DEADLINE - 30, seed=seed * 19 + 7,
                initial_temp=max(10.0, best_scope * 0.2))
            if _is_valid(sa_rows) and sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
    
    # Phase 4: Local search to reduce scope further
    if time.time() < DEADLINE - 15:
        for seed in range(3):
            if time.time() > DEADLINE - 12:
                break
            ls_rows, ls_scope = _local_search_refine(
                best_rows, k, DEADLINE - 10, seed=seed * 23 + 13)
            if _is_valid(ls_rows) and ls_scope < best_scope:
                best_scope = ls_scope
                best_rows = ls_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        rows2 = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows2.append(row)
        return rows2


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
