"""Difference Triangle Set (4,7) solver.

Strategy:
1. Fast greedy baseline with many randomized variants.
2. Local compaction (right-to-left greedy minimization of each element).
3. MRV backtracking search that tries to place each element at its smallest
   feasible value, pruning branches that cannot lead to a valid solution.
4. Time-bounded simulated annealing as a fallback for further improvement.
5. Safety margin before the 150 s wall-clock limit.
"""
import time
import random
import sys

sys.setrecursionlimit(10000)


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


def _greedy_variant(n_rows, k, rng):
    """Randomized greedy with variable skip probability."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        skip_p = rng.random() * 0.15
        while len(row) < k + 1:
            x += 1
            if rng.random() < skip_p:
                x += 1
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
        if len(row) != k + 1:
            return None
        rows.append(row)
    return rows


def _build_used_set(rows):
    """Build the set of all positive differences across all rows."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                used.add(row[j] - row[i])
    return used


def _get_diffs_for_pos(row, pos):
    """Get all differences involving position pos in a row."""
    L = len(row)
    diffs = set()
    for i in range(L):
        if i < pos:
            diffs.add(row[pos] - row[i])
        elif i > pos:
            diffs.add(row[i] - row[pos])
    return diffs


def _can_set(row, pos, new_val, used_set):
    """Check if we can set row[pos] to new_val without creating conflicts."""
    L = len(row)
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < L - 1 and new_val >= row[pos + 1]:
        return False
    old_diffs = _get_diffs_for_pos(row, pos)
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


def _apply_set(row, pos, new_val, used_set):
    """Apply setting row[pos] to new_val, updating used_set."""
    old_diffs = _get_diffs_for_pos(row, pos)
    new_diffs = set()
    L = len(row)
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


def _fast_compact_row(row, used_set, time_limit):
    """Compact a single row from right to left, setting each element to its
    smallest feasible value."""
    L = len(row)
    for pos in range(L - 1, 0, -1):
        if time.time() > time_limit:
            break
        old_val = row[pos]
        lower = row[pos - 1] + 1
        if lower >= old_val:
            continue
        old_diffs = _get_diffs_for_pos(row, pos)
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
            break


def _local_compact(rows, n_rows, k, time_limit):
    """Iteratively compact all rows until no more improvements."""
    work = [list(r) for r in rows]
    used_set = _build_used_set(work)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for r_idx in range(n_rows):
            old_max = max(work[r_idx])
            _fast_compact_row(work[r_idx], used_set, time_limit)
            if max(work[r_idx]) < old_max:
                improved = True
    return work


def _mrv_backtrack(rows, n_rows, k, time_limit, target_max):
    """MRV backtracking search to find a valid solution with max entry <= target_max.
    
    Tries to place each element at its smallest feasible value, using MRV
    (Minimum Remaining Values) to select the next position to fix.
    """
    work = [list(r) for r in rows]
    L = k + 1
    used_set = _build_used_set(work)
    current_max = _max_entry(work)
    
    if current_max <= target_max:
        return work
    
    # We need to reduce the maximum entry. Find which rows have entries > target_max
    # and try to compact them.
    
    def try_compact_rows(remaining_rows, used_set_local, time_limit_local):
        """Try to compact the remaining rows. Returns True if all rows are valid."""
        if not remaining_rows:
            return True
        if time.time() > time_limit_local:
            return False
        
        r_idx = remaining_rows[0]
        row = work[r_idx]
        
        # Find the largest element in this row that exceeds target_max
        # and try to reduce it
        for pos in range(L - 1, 0, -1):
            if row[pos] <= target_max:
                continue
            
            old_val = row[pos]
            lower = row[pos - 1] + 1
            old_diffs = _get_diffs_for_pos(row, pos)
            
            # Try values from lower up to min(old_val-1, target_max)
            upper = min(old_val - 1, target_max)
            found = False
            for new_val in range(lower, upper + 1):
                if pos < L - 1 and new_val >= row[pos + 1]:
                    break
                
                new_diffs = set()
                for i in range(L):
                    if i < pos:
                        new_diffs.add(new_val - row[i])
                    elif i > pos:
                        new_diffs.add(row[i] - new_val)
                
                other = used_set_local - old_diffs
                if new_diffs & other:
                    continue
                
                # This value works, try it
                for d in old_diffs:
                    used_set_local.discard(d)
                for d in new_diffs:
                    used_set_local.add(d)
                row[pos] = new_val
                
                # Recurse on remaining rows
                if try_compact_rows(remaining_rows[1:], used_set_local, time_limit_local):
                    return True
                
                # Backtrack
                for d in new_diffs:
                    used_set_local.discard(d)
                for d in old_diffs:
                    used_set_local.add(d)
                row[pos] = old_val
                found = True
                break
            
            if not found:
                # Can't reduce this position, try next
                continue
        
        # If we couldn't reduce any position in this row that exceeds target_max,
        # the row is already as compact as possible
        return try_compact_rows(remaining_rows[1:], used_set_local, time_limit_local)
    
    # Sort rows by their current max entry (descending) - most constrained first
    row_order = sorted(range(n_rows), key=lambda i: -max(work[i]))
    
    if try_compact_rows(row_order, used_set, time_limit):
        if _check_valid(work, k) and _max_entry(work) <= target_max:
            return work
    
    return None


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = _build_used_set(work)
    
    def recompact_row(r_idx):
        _fast_compact_row(work[r_idx], used_set, time_limit)
    
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max
    
    start_temp = max(1, current_max // 10)
    end_temp = 0.1
    cooling = 0.9995
    
    t = start_temp
    step = 0
    
    while time.time() < time_limit and t > end_temp:
        step += 1
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to decrease a random element
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                continue
            new_val = rng.randint(lower, old_val - 1)
            if new_val >= old_val:
                continue
            if _can_set(row, pos, new_val, used_set):
                _apply_set(row, pos, new_val, used_set)
                recompact_row(r_idx)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    
        elif move_type < 0.8:
            # Increase a random element, then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            new_val = old_val + rng.randint(1, max(1, current_max // 5))
            if _can_set(row, pos, new_val, used_set):
                _apply_set(row, pos, new_val, used_set)
                recompact_row(r_idx)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
            else:
                new_val = old_val + 1
                if _can_set(row, pos, new_val, used_set):
                    _apply_set(row, pos, new_val, used_set)
                    recompact_row(r_idx)
                    new_max = _max_entry(work)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in work]
        else:
            # Full recompact of a random row
            r_idx = rng.randint(0, n_rows - 1)
            recompact_row(r_idx)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
        
        t *= cooling
        
        if step % 200 == 0:
            for r_idx in range(n_rows):
                recompact_row(r_idx)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
    
    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 135  # Safety margin

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Randomized greedy restarts
    rng = random.Random(42)
    for trial in range(100):
        if time.time() > time_limit - 60:
            break
        rows = _greedy_variant(n_rows, k, rng)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Local compaction
    compacted = _local_compact(best_rows, n_rows, k, time_limit - 30)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax

    # 4) MRV backtracking to try to reach a tighter bound
    if time.time() < time_limit - 20:
        # Try to find a solution with max entry = best_max - 1
        target = best_max - 1
        mrv_result = _mrv_backtrack(best_rows, n_rows, k, time_limit - 15, target)
        if mrv_result is not None:
            best_rows = mrv_result
            best_max = _max_entry(best_rows)

    # 5) Simulated annealing with remaining time
    if time.time() < time_limit - 5:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 5)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 6) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
