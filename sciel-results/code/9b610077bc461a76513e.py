"""Difference Triangle Set (4,7) solver.

Combined strategy:
- Fast greedy baseline + many randomized greedy variants (with row-order shuffling).
- Efficient local compaction (right-to-left, set-based difference checks).
- MRV-based backtracking CSP solver that selects next variable by most-constrained
  difference slots rather than sequential position.
- Time-bounded simulated annealing with:
    * single-element decrease (compaction) moves
    * random element perturbations + recompact
    * full-row rebuilds (multiple attempts, bounded by current max)
    * full recompact passes
- Safety margin before the 150 s wall-clock limit.
"""
import time
import random
import numpy as np
from collections import defaultdict


def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, rng=None):
    """Greedy ruler construction with optional randomization."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < 0.02:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _greedy_variant(n_rows, k, rng, skip_p=0.05, jump_p=0.03, shuffle_rows=False, bias_p=0.0, bias_range=5):
    """Greedy with configurable skip and jump probabilities, optional row-order shuffle, and bias."""
    used = set()
    rows = []
    # Optionally shuffle the order in which rows are built
    order = list(range(n_rows))
    if shuffle_rows:
        rng.shuffle(order)
    row_results = {}
    for r_idx in order:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < jump_p:
                x += rng.randint(1, 4)
            if rng.random() < skip_p:
                continue
            # Bias: sometimes try a larger jump
            if bias_p > 0 and rng.random() < bias_p:
                x += rng.randint(1, bias_range)
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        if len(row) != k + 1:
            return None
        row_results[r_idx] = row
    for r_idx in range(n_rows):
        rows.append(row_results[r_idx])
    return rows


def _build_used_set(rows, k):
    L = k + 1
    used = set()
    for row in rows:
        for i in range(L):
            for j in range(i + 1, L):
                used.add(row[j] - row[i])
    return used


def _get_diffs_for_pos(row, pos, L):
    diffs = set()
    for i in range(L):
        if i < pos:
            diffs.add(row[pos] - row[i])
        elif i > pos:
            diffs.add(row[i] - row[pos])
    return diffs


def _can_set(row, pos, new_val, used_set, L):
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < L - 1 and new_val >= row[pos + 1]:
        return False
    old_diffs = _get_diffs_for_pos(row, pos, L)
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    other = used_set - old_diffs
    if new_diffs & other:
        return False
    return True


def _apply_set(row, pos, new_val, used_set, L):
    old_diffs = _get_diffs_for_pos(row, pos, L)
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    for d in old_diffs:
        used_set.discard(d)
    for d in new_diffs:
        used_set.add(d)
    row[pos] = new_val


def _fast_compact_row(row, used_set, L):
    """Compact a single row in-place, return True if any change made."""
    changed = False
    for pos in range(L - 1, 0, -1):
        old_val = row[pos]
        lower = row[pos - 1] + 1
        if lower >= old_val:
            continue
        old_diffs = _get_diffs_for_pos(row, pos, L)
        for new_val in range(lower, old_val):
            if pos < L - 1 and new_val >= row[pos + 1]:
                break
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            other = used_set - old_diffs
            if new_diffs & other:
                continue
            for d in old_diffs:
                used_set.discard(d)
            for d in new_diffs:
                used_set.add(d)
            row[pos] = new_val
            changed = True
            break
    return changed


def _local_compact(rows, n_rows, k, time_limit):
    """Iteratively compact each element to its smallest feasible value."""
    work = [list(r) for r in rows]
    used_set = _build_used_set(work, k)
    L = k + 1
    improved = True

    while improved and time.time() < time_limit:
        improved = False
        for r_idx in range(n_rows):
            row = work[r_idx]
            for pos in range(L - 1, 0, -1):
                old_val = row[pos]
                lower = row[pos - 1] + 1
                if lower >= old_val:
                    continue
                old_diffs = _get_diffs_for_pos(row, pos, L)
                for new_val in range(lower, old_val):
                    if pos < L - 1 and new_val >= row[pos + 1]:
                        break
                    new_diffs = set()
                    for i in range(L):
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        elif i > pos:
                            new_diffs.add(row[i] - new_val)
                    other = used_set - old_diffs
                    if new_diffs & other:
                        continue
                    for d in old_diffs:
                        used_set.discard(d)
                    for d in new_diffs:
                        used_set.add(d)
                    row[pos] = new_val
                    improved = True
                    break
    return work


def _rebuild_row(r_idx, work, used_set, L, rng, max_val):
    """Rebuild a row from scratch, trying to keep its max below max_val."""
    row = work[r_idx]
    old_diffs = set()
    for i in range(L):
        for j in range(i + 1, L):
            old_diffs.add(row[j] - row[i])
    for d in old_diffs:
        used_set.discard(d)

    best_new_row = None
    best_new_max = max_val + 1

    # Try multiple strategies
    for attempt in range(50):
        strategy = rng.random()
        new_row = [0]
        x = 0
        valid = True
        
        if strategy < 0.4:
            # Standard greedy
            while len(new_row) < L:
                x += 1
                if rng.random() < 0.05:
                    x += rng.randint(1, 3)
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break
        elif strategy < 0.7:
            # Aggressive jumps
            while len(new_row) < L:
                x += rng.randint(1, 4)
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break
        else:
            # Conservative with occasional jumps
            while len(new_row) < L:
                x += 1
                if rng.random() < 0.15:
                    x += rng.randint(2, 5)
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break
        
        if valid and len(new_row) == L:
            new_max = new_row[-1]
            if new_max < best_new_max:
                best_new_max = new_max
                best_new_row = list(new_row)

    if best_new_row is not None:
        # Add new row's diffs to used_set
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(best_new_row[j] - best_new_row[i])
        work[r_idx] = best_new_row
        return True
    else:
        # Restore old row
        for d in old_diffs:
            used_set.add(d)
        work[r_idx] = row
        return False


def _mrv_backtrack(n_rows, k, rng, time_limit, max_bound=None):
    """MRV-based backtracking CSP solver.
    
    Variables: (row_idx, pos) for pos in 1..k, row_idx in 0..n_rows-1
    Domain: integers that maintain strict increase and unique differences.
    Ordering: MRV - pick variable with fewest feasible values given current assignments.
    
    Returns best valid solution found within time_limit, or None.
    """
    L = k + 1
    total_vars = n_rows * k  # positions 1..k in each row
    
    # State: rows as lists, used_diffs set
    rows = [[0] * L for _ in range(n_rows)]
    used_diffs = set()
    
    # Track assigned positions
    assigned = [[False] * L for _ in range(n_rows)]
    for r in range(n_rows):
        assigned[r][0] = True  # position 0 is always 0
    
    best_solution = None
    best_max = max_bound if max_bound else float('inf')
    start_time = time.time()
    
    def get_domain(r, pos, current_rows, current_used):
        """Get feasible values for position pos in row r."""
        if pos == 0:
            return [0]
        
        lower = current_rows[r][pos-1] + 1 if pos > 0 and assigned[r][pos-1] else 1
        upper = current_rows[r][pos+1] - 1 if pos < L-1 and assigned[r][pos+1] else (max_bound - 1 if max_bound else 200)
        
        if lower >= upper:
            return []
        
        feasible = []
        for val in range(lower, min(upper, (max_bound if max_bound else 200) + 1)):
            # Check differences with all other assigned positions in this row
            ok = True
            for i in range(L):
                if i == pos or not assigned[r][i]:
                    continue
                d = abs(val - current_rows[r][i])
                if d in current_used:
                    ok = False
                    break
            if ok:
                feasible.append(val)
        
        return feasible
    
    def count_domain_size(r, pos, current_rows, current_used):
        """Count feasible values without generating them (faster for MRV)."""
        if pos == 0:
            return 1
        
        lower = current_rows[r][pos-1] + 1 if pos > 0 and assigned[r][pos-1] else 1
        upper = current_rows[r][pos+1] - 1 if pos < L-1 and assigned[r][pos+1] else (max_bound - 1 if max_bound else 200)
        
        if lower >= upper:
            return 0
        
        count = 0
        for val in range(lower, min(upper, (max_bound if max_bound else 200) + 1)):
            ok = True
            for i in range(L):
                if i == pos or not assigned[r][i]:
                    continue
                d = abs(val - current_rows[r][i])
                if d in current_used:
                    ok = False
                    break
            if ok:
                count += 1
        
        return count
    
    def backtrack(depth, current_rows, current_used):
        nonlocal best_solution, best_max, start_time
        
        if time.time() - start_time > time_limit:
            return False
        
        if depth == total_vars:
            # All variables assigned
            row_max = max(max(r) for r in current_rows)
            if row_max < best_max:
                best_max = row_max
                best_solution = [list(r) for r in current_rows]
            return True
        
        # MRV: find unassigned variable with minimum domain size
        min_size = float('inf')
        best_r, best_pos = -1, -1
        
        for r in range(n_rows):
            for pos in range(1, L):
                if assigned[r][pos]:
                    continue
                size = count_domain_size(r, pos, current_rows, current_used)
                if size == 0:
                    return False  # Dead end
                if size < min_size:
                    min_size = size
                    best_r, best_pos = r, pos
                    if size == 1:
                        break
            if min_size == 1:
                break
        
        if best_r == -1:
            return False
        
        # Get domain for the selected variable
        domain = get_domain(best_r, best_pos, current_rows, current_used)
        
        if not domain:
            return False
        
        # Randomize domain order for diversity
        rng.shuffle(domain)
        
        # Try values in order
        for val in domain:
            if time.time() - start_time > time_limit:
                return False
            
            # Assign
            current_rows[best_r][best_pos] = val
            assigned[best_r][best_pos] = True
            
            # Add differences
            new_diffs = set()
            for i in range(L):
                if i != best_pos and assigned[best_r][i]:
                    d = abs(val - current_rows[best_r][i])
                    new_diffs.add(d)
            
            conflict = False
            for d in new_diffs:
                if d in current_used:
                    conflict = True
                    break
            
            if not conflict:
                current_used.update(new_diffs)
                
                if backtrack(depth + 1, current_rows, current_used):
                    return True
                
                current_used.difference_update(new_diffs)
            
            # Unassign
            assigned[best_r][best_pos] = False
            current_rows[best_r][best_pos] = 0
        
        return False
    
    # Run multiple restarts with different random seeds
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Reset state
        rows = [[0] * L for _ in range(n_rows)]
        used_diffs = set()
        assigned = [[False] * L for _ in range(n_rows)]
        for r in range(n_rows):
            assigned[r][0] = True
        
        # Use current best_max as bound
        effective_bound = best_max - 1 if best_max < float('inf') else None
        
        if backtrack(0, rows, used_diffs):
            if best_solution is not None:
                return best_solution
        
        # If we found a solution, tighten the bound and restart
        if best_solution is not None:
            best_max = max(max(r) for r in best_solution)
    
    return best_solution


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = _build_used_set(work, k)
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max

    start_temp = max(1, current_max // 8)
    end_temp = 0.05
    cooling = 0.9995

    t = start_temp
    stagnation_counter = 0
    max_stagnation = 2000

    while time.time() < time_limit and t > end_temp:
        move_type = rng.random()

        if move_type < 0.35:
            # Move 1: Try to decrease a random element
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                t *= cooling
                continue
            new_val = rng.randint(lower, old_val - 1)
            if new_val >= old_val:
                t *= cooling
                continue
            if _can_set(row, pos, new_val, used_set, L):
                _apply_set(row, pos, new_val, used_set, L)
                _fast_compact_row(row, used_set, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

        elif move_type < 0.65:
            # Move 2: Increase a random element (perturbation), then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(2, current_max // 3))
            new_val = rng.randint(old_val + 1, upper_limit)
            if new_val <= old_val:
                t *= cooling
                continue
            if _can_set(row, pos, new_val, used_set, L):
                _apply_set(row, pos, new_val, used_set, L)
                _fast_compact_row(row, used_set, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                # Try smaller increments
                for delta in range(1, 5):
                    candidate = old_val + delta
                    if _can_set(row, pos, candidate, used_set, L):
                        _apply_set(row, pos, candidate, used_set, L)
                        _fast_compact_row(row, used_set, L)
                        new_max = _max_entry(work)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(r) for r in work]
                            current_max = best_max
                            stagnation_counter = 0
                        break

        elif move_type < 0.80:
            # Move 3: Rebuild a random row with more attempts
            r_idx = rng.randint(0, n_rows - 1)
            _rebuild_row(r_idx, work, used_set, L, rng, current_max)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        elif move_type < 0.90:
            # Move 4: Full recompact of all rows
            for r_idx in range(n_rows):
                _fast_compact_row(work[r_idx], used_set, L)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        else:
            # Move 5: Swap two elements between different rows (if valid)
            r1 = rng.randint(0, n_rows - 1)
            r2 = rng.randint(0, n_rows - 1)
            if r1 != r2:
                p1 = rng.randint(1, L - 1)
                p2 = rng.randint(1, L - 1)
                row1 = work[r1]
                row2 = work[r2]
                v1 = row1[p1]
                v2 = row2[p2]
                
                # Check if we can swap
                old_diffs1 = _get_diffs_for_pos(row1, p1, L)
                old_diffs2 = _get_diffs_for_pos(row2, p2, L)
                
                # Check bounds for row1
                if p1 > 0 and v2 <= row1[p1-1]:
                    pass
                elif p1 < L-1 and v2 >= row1[p1+1]:
                    pass
                else:
                    # Check bounds for row2
                    if p2 > 0 and v1 <= row2[p2-1]:
                        pass
                    elif p2 < L-1 and v1 >= row2[p2+1]:
                        pass
                    else:
                        # Try the swap
                        new_diffs1 = set()
                        for i in range(L):
                            if i < p1:
                                new_diffs1.add(v2 - row1[i])
                            elif i > p1:
                                new_diffs1.add(row1[i] - v2)
                        
                        new_diffs2 = set()
                        for i in range(L):
                            if i < p2:
                                new_diffs2.add(v1 - row2[i])
                            elif i > p2:
                                new_diffs2.add(row2[i] - v1)
                        
                        other = used_set - old_diffs1 - old_diffs2
                        if not (new_diffs1 & other) and not (new_diffs2 & other) and not (new_diffs1 & new_diffs2):
                            # Apply swap
                            for d in old_diffs1:
                                used_set.discard(d)
                            for d in old_diffs2:
                                used_set.discard(d)
                            for d in new_diffs1:
                                used_set.add(d)
                            for d in new_diffs2:
                                used_set.add(d)
                            row1[p1] = v2
                            row2[p2] = v1
                            _fast_compact_row(row1, used_set, L)
                            _fast_compact_row(row2, used_set, L)
                            new_max = _max_entry(work)
                            if new_max < best_max:
                                best_max = new_max
                                best_rows = [list(r) for r in work]
                                current_max = best_max
                                stagnation_counter = 0
                            else:
                                stagnation_counter += 1

        # If stuck for too long, restart with a fresh greedy
        if stagnation_counter > max_stagnation:
            fresh = _greedy_variant(n_rows, k, rng, skip_p=rng.random()*0.15, 
                                    jump_p=rng.random()*0.08, shuffle_rows=True,
                                    bias_p=rng.random()*0.3, bias_range=rng.randint(3,8))
            if fresh is not None and _check_valid(fresh, k):
                fresh_max = _max_entry(fresh)
                if fresh_max < best_max:
                    best_max = fresh_max
                    best_rows = [list(r) for r in fresh]
                    current_max = best_max
                    work = [list(r) for r in fresh]
                    used_set = _build_used_set(work, k)
                    t = start_temp
                    stagnation_counter = 0

        t *= cooling

    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: stop searching at 140 s to leave room for final validation
    time_limit = start_time + 140

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts with varied parameters
    rng = random.Random(42)
    greedy_deadline = time_limit - 80
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        # More diverse parameter ranges
        skip_p = rng.random() * 0.15
        jump_p = rng.random() * 0.08
        shuffle = rng.random() < 0.6
        bias_p = rng.random() * 0.4
        bias_range = rng.randint(3, 8)
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                               shuffle_rows=shuffle, bias_p=bias_p, bias_range=bias_range)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) MRV-based backtracking CSP solver (structural change)
    mrv_deadline = time_limit - 50
    if time.time() < mrv_deadline:
        mrv_result = _mrv_backtrack(n_rows, k, rng, mrv_deadline - time.time(), max_bound=best_max)
        if mrv_result is not None and _check_valid(mrv_result, k):
            mmax = _max_entry(mrv_result)
            if mmax < best_max:
                best_rows = mrv_result
                best_max = mmax

    # 4) Local compaction
    compacted = _local_compact(best_rows, n_rows, k, time_limit - 25)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax

    # 5) Simulated annealing with remaining time
    if time.time() < time_limit - 3:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 3)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 6) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
