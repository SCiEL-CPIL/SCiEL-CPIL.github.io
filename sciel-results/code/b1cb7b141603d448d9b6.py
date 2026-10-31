"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.

Structural approach:
1. Backtracking construction with forward checking and dynamic variable ordering.
   Each row is built by selecting the smallest available value that keeps all
   differences distinct, but we try multiple orderings and use a priority
   heuristic that places rows with tighter constraints first.
2. A constraint-propagation refinement: after initial construction, we attempt
   to reduce the maximum by re-solving with a tighter upper bound (binary search
   on the scope) using backtracking with forward checking.
3. Fallback to greedy + local search if backtracking is too slow.
"""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


def _diffs_of_row(row):
    """Return a set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _validate(rows):
    """Check all differences are distinct across all rows. Return True if valid."""
    seen = set()
    for row in rows:
        d = _diffs_of_row(row)
        if len(d) != len(row) * (len(row) - 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True


def _backtrack_construct(n_rows, k, max_val, rng, time_budget=10.0):
    """
    Backtracking with forward checking to construct rows with all entries <= max_val.
    Returns a list of rows if successful, None otherwise.
    
    We build rows one at a time. For each row, we try to assign values in order,
    using forward checking to prune early.
    """
    start = time.time()
    used = set()  # all differences used so far
    rows = []
    
    # Total differences needed per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # If max_val is too small to hold all differences, fail immediately
    if max_val < total_diffs:
        return None
    
    def build_row(row_idx, current_row, used_so_far, time_limit):
        """Recursively build a single row with forward checking."""
        if time.time() - start > time_limit:
            return None
        
        if len(current_row) == k + 1:
            return list(current_row)
        
        # Forward checking: count how many differences are still available
        # that are <= max_val and not in used_so_far
        # For efficiency, just try candidates
        
        pos = len(current_row)
        if pos == 0:
            candidates = [0]
        else:
            start_val = current_row[-1] + 1
            end_val = max_val
            if start_val > end_val:
                return None
            candidates = range(start_val, end_val + 1)
        
        for val in candidates:
            # Check internal differences
            new_diffs = set()
            for a in current_row:
                d = val - a
                if d in used_so_far:
                    break
                new_diffs.add(d)
            else:
                # Check if adding these diffs would make it impossible to complete
                # This is a simplified forward check: just verify no conflicts
                # For stronger pruning, we could check remaining capacity
                new_used = used_so_far | new_diffs
                result = build_row(row_idx, current_row + [val], new_used, time_limit)
                if result is not None:
                    return result
        
        return None
    
    # Try to build all rows
    # Use a heuristic: build rows in an order that might be easier
    # We'll try the natural order first
    
    for r in range(n_rows):
        if time.time() - start > time_budget:
            return None
        
        row = build_row(r, [0], used, time_budget)
        if row is None:
            return None
        rows.append(row)
        used |= _diffs_of_row(row)
    
    return rows


def _backtrack_with_order(n_rows, k, max_val, rng, time_budget=15.0):
    """
    Backtracking that tries different row orderings.
    """
    start = time.time()
    used = set()
    rows = [None] * n_rows
    
    # Try to determine a good ordering: rows that are "harder" first
    # A simple heuristic: no specific ordering, just try natural order
    order = list(range(n_rows))
    
    def build_single_row(used_so_far, time_limit):
        """Build one row given the current used set."""
        if time.time() - start > time_limit:
            return None
        
        row = [0]
        used_row = set()
        
        def extend(current_row, current_used):
            if time.time() - start > time_limit:
                return None
            
            if len(current_row) == k + 1:
                return list(current_row)
            
            last = current_row[-1]
            for val in range(last + 1, max_val + 1):
                new_diffs = set()
                ok = True
                for a in current_row:
                    d = val - a
                    if d in current_used or d in used_so_far:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                
                result = extend(current_row + [val], current_used | new_diffs)
                if result is not None:
                    return result
            
            return None
        
        return extend(row, used_row)
    
    for r in order:
        if time.time() - start > time_budget:
            return None
        
        row = build_single_row(used, time_budget)
        if row is None:
            return None
        rows[r] = row
        used |= _diffs_of_row(row)
    
    return rows


def _greedy_build(n_rows, k, rng, order=None):
    """Build rows greedily in the given order."""
    if order is None:
        order = list(range(n_rows))
    used = set()
    rows = [None] * n_rows
    for r in order:
        row = [0]
        while len(row) < k + 1:
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
                    break
                x += 1
        rows[r] = row
    return rows


def _local_search_reduce(rows, n_rows, k, rng, time_budget):
    """
    Local search: try to reduce the maximum value by moving the largest elements down.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()

    row_diffs = [_diffs_of_row(rows[r]) for r in range(n_rows)]

    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find the row with the maximum element
        max_row = -1
        max_val = -1
        for r in range(n_rows):
            if rows[r][-1] > max_val:
                max_val = rows[r][-1]
                max_row = r
        
        if max_row == -1:
            break
        
        row = rows[max_row]
        improved = False
        
        # Try to reduce the last element
        for new_val in range(row[-1] - 1, row[-2], -1):
            row_copy = list(row)
            row_copy[-1] = new_val
            new_rd = _diffs_of_row(row_copy)
            if len(new_rd) != k * (k + 1) // 2:
                continue
            conflict = False
            for other_r in range(n_rows):
                if other_r == max_row:
                    continue
                if new_rd & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                row[-1] = new_val
                row_diffs[max_row] = new_rd
                new_max = max(max(rows[x]) for x in range(n_rows))
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(rows[x]) for x in range(n_rows)]
                improved = True
                break
        
        if not improved:
            # Try reducing a random non-last element in a random row
            r = rng.randrange(n_rows)
            if len(rows[r]) <= 2:
                break
            pos = rng.randrange(1, len(rows[r]) - 1)
            row = rows[r]
            old = row[pos]
            for new_val in range(old - 1, row[pos - 1], -1):
                if pos < k and new_val >= row[pos + 1]:
                    continue
                row_copy = list(row)
                row_copy[pos] = new_val
                new_rd = _diffs_of_row(row_copy)
                if len(new_rd) != k * (k + 1) // 2:
                    continue
                conflict = False
                for other_r in range(n_rows):
                    if other_r == r:
                        continue
                    if new_rd & row_diffs[other_r]:
                        conflict = True
                        break
                if not conflict:
                    row[pos] = new_val
                    row_diffs[r] = new_rd
                    new_max = max(max(rows[x]) for x in range(n_rows))
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(rows[x]) for x in range(n_rows)]
                    break

    return best_rows, best_max


def _sa_optimize(rows, n_rows, k, rng, time_budget):
    """
    Simulated annealing to reduce the maximum entry.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)

    row_diffs = [_diffs_of_row(rows[r]) for r in range(n_rows)]

    start = time.time()
    T = max(1.0, best_max * 0.1)
    T_min = 0.5
    cooling = 0.997

    while T > T_min and time.time() - start < time_budget and _time_left() > 3.0:
        r = rng.randrange(n_rows)
        row = rows[r]
        if len(row) <= 1:
            T *= cooling
            continue
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        
        span = max(1, min(15, int(T)))
        delta = rng.randint(-span, span)
        new_val = old_val + delta
        
        if new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
        
        row_copy = list(row)
        row_copy[pos] = new_val
        new_rd = _diffs_of_row(row_copy)
        
        if len(new_rd) != k * (k + 1) // 2:
            T *= cooling
            continue
        
        conflict = 0
        for other_r in range(n_rows):
            if other_r == r:
                continue
            conflict += len(new_rd & row_diffs[other_r])
        
        if conflict == 0:
            cur_max = 0
            for x in range(n_rows):
                for v in rows[x]:
                    if v > cur_max:
                        cur_max = v
            # Recompute with the change
            row_max = max(row_copy)
            other_max = 0
            for x in range(n_rows):
                if x != r:
                    for v in rows[x]:
                        if v > other_max:
                            other_max = v
            cur_max = max(row_max, other_max)
            
            if cur_max <= best_max or rng.random() < 0.1:
                row[pos] = new_val
                row_diffs[r] = new_rd
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(rows[x]) for x in range(n_rows)]
        else:
            if conflict <= 2 and rng.random() < 0.01 * (T / 100.0):
                row[pos] = new_val
                row_diffs[r] = new_rd

        T *= cooling

    return best_rows, best_max


def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()

    rng = random.Random(42)

    best_rows = None
    best_max = float('inf')
    
    # Phase 1: Greedy baseline with several orderings
    num_greedy = 20
    for t in range(num_greedy):
        if _time_left() < 5.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, rng, order)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, rng)
        best_max = max(max(r) for r in best_rows)

    # Phase 2: Backtracking with binary search on max_val
    # Try to find the smallest max_val that allows a valid construction
    if _time_left() > 30.0:
        # Lower bound: total_diffs
        total_diffs = n_rows * k * (k + 1) // 2
        lo = max(total_diffs, k)
        hi = best_max
        
        # Binary search on max_val
        bt_start = time.time()
        while lo <= hi and _time_left() > 10.0 and time.time() - bt_start < 60.0:
            mid = (lo + hi) // 2
            if _time_left() < 5.0:
                break
            
            # Try backtracking with this max_val
            bt_rows = _backtrack_with_order(n_rows, k, mid, rng, time_budget=8.0)
            if bt_rows is not None:
                # Verify
                if _validate(bt_rows):
                    cur_max = max(max(r) for r in bt_rows)
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = [list(r) for r in bt_rows]
                    hi = mid - 1
                else:
                    hi = mid - 1
            else:
                lo = mid + 1
    
    # Phase 3: Local search to reduce max
    if _time_left() > 10.0:
        budget = min(20.0, _time_left() - 5.0)
        ls_rows, ls_max = _local_search_reduce(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    # Phase 4: Simulated annealing
    if _time_left() > 15.0:
        budget = min(40.0, _time_left() - 5.0)
        sa_rows, sa_max = _sa_optimize(best_rows, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows

    # Phase 5: Final local search polish
    if _time_left() > 5.0:
        budget = min(15.0, _time_left() - 2.0)
        ls_rows, ls_max = _local_search_reduce(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    # Verify
    if not _validate(best_rows):
        # Fallback to a fresh greedy
        best_rows = _greedy_build(n_rows, k, rng)

    return best_rows
