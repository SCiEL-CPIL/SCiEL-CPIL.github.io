"""
Difference-triangle-set (4,7): construct 4 rows of 8 increasing integers starting at 0
so that all positive differences within rows are globally distinct. Minimize the largest entry.

Strategy:
- Use a randomized greedy construction with a bitset for O(1) difference-availability checks.
- Perform multiple randomized restarts with different row orderings and perturbations.
- Apply local search (hill-climbing) to reduce the maximum entry.
- Use a backtracking approach for small subproblems if greedy gets stuck.
- Explicit deadline guard to return best result found before the 150s limit.
"""
import time
import random
import sys

_DEADLINE = 150.0
_MARGIN = 15.0  # safety margin (seconds)


def _compute_diffs(row):
    """Return set of all positive differences within a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows):
    """Check that all positive differences are distinct across all rows."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_build_row(row_start, used_diffs, k, max_val):
    """
    Greedily extend a row (list, starts with 0) to k+1 marks.
    used_diffs: set of already-used differences.
    Returns the completed row list, or None if stuck.
    """
    row = list(row_start)
    n_marks = len(row)
    while n_marks < k + 1:
        found = False
        for x in range(row[-1] + 1, max_val + 1):
            conflict = False
            for a in row:
                d = x - a
                if d > max_val or d in used_diffs:
                    conflict = True
                    break
            if not conflict:
                for a in row:
                    used_diffs.add(x - a)
                row.append(x)
                n_marks += 1
                found = True
                break
        if not found:
            return None
    return row


def _greedy_build_all(n_rows, k, max_val, rng):
    """
    Build n_rows rows greedily with a random row ordering.
    Returns list of rows or None.
    """
    used = set()
    rows = []
    order = list(range(n_rows))
    rng.shuffle(order)
    for idx in order:
        row = _greedy_build_row([0], used, k, max_val)
        if row is None:
            return None
        rows.append(row)
    return rows


def _hill_climb(rows, k, time_limit, rng):
    """
    Try to reduce the maximum entry by moving individual marks to smaller valid positions.
    Returns (improved_rows, best_max).
    """
    rows = [list(r) for r in rows]
    best_max = _max_entry(rows)
    n_rows = len(rows)
    k_plus_1 = k + 1

    def compute_used():
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    used.add(row[j] - row[i])
        return used

    used = compute_used()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try moving each mark in each row to a smaller valid position
        for r in range(n_rows):
            row = rows[r]
            for pos in range(1, k_plus_1):
                old_val = row[pos]
                # Try smaller values in random order for diversity
                candidates = list(range(row[pos - 1] + 1, old_val))
                rng.shuffle(candidates)
                for new_val in candidates:
                    tent = list(row)
                    tent[pos] = new_val
                    # Check strictly increasing
                    valid = True
                    for i in range(len(tent) - 1):
                        if tent[i + 1] <= tent[i]:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Compute diffs of tentative row
                    tent_diffs = _compute_diffs(tent)
                    # Compute old diffs
                    old_diffs = _compute_diffs(row)
                    other_diffs = used - old_diffs
                    if tent_diffs & other_diffs:
                        continue
                    # Accept the move
                    used = (used - old_diffs) | tent_diffs
                    rows[r] = tent
                    if new_val < best_max:
                        best_max = new_val
                        improved = True
                    break  # move to next position
    return rows, best_max


def _backtrack_build_row(row_start, used_diffs, k, max_val, deadline):
    """
    Backtracking search to extend a row. More thorough than greedy but slower.
    Returns the completed row list, or None if stuck or time exceeded.
    """
    row = list(row_start)
    n_marks = len(row)
    
    def backtrack(current_row, current_used):
        if time.time() > deadline:
            return None
        if len(current_row) == k + 1:
            return list(current_row)
        last = current_row[-1]
        for x in range(last + 1, max_val + 1):
            conflict = False
            new_diffs = []
            for a in current_row:
                d = x - a
                if d > max_val or d in current_used:
                    conflict = True
                    break
                new_diffs.append(d)
            if conflict:
                continue
            # Try placing x
            current_row.append(x)
            current_used.update(new_diffs)
            result = backtrack(current_row, current_used)
            if result is not None:
                return result
            current_row.pop()
            current_used.difference_update(new_diffs)
        return None
    
    return backtrack(row, set(used_diffs))


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all positive differences distinct.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN

    # Estimate a reasonable max_val
    total_diffs_needed = n_rows * k * (k + 1) // 2
    # Known best for (4,7) is 134; counting bound is 112
    initial_max = max(total_diffs_needed, 135)

    best_rows = None
    best_max = initial_max

    rng = random.Random(12345)

    # Phase 1: Multiple greedy constructions with different random seeds/orderings
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        max_val = initial_max + rng.randint(0, 30)
        rows = _greedy_build_all(n_rows, k, max_val, rng)
        if rows is not None:
            cur_max = _max_entry(rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in rows]
        # If we've tried many times and not improving, break
        if attempts > 200 and best_rows is not None:
            break

    # Phase 2: Hill-climbing on the best solution found
    if best_rows is not None:
        if time.time() < deadline:
            rows, cur_max = _hill_climb(best_rows, k, deadline, rng)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = rows

    # Phase 3: More aggressive search with backtracking for individual rows
    if best_rows is not None and time.time() < deadline:
        # Try rebuilding each row with backtracking to potentially find better solutions
        for r in range(n_rows):
            if time.time() > deadline:
                break
            # Compute used diffs from other rows
            other_used = set()
            for i, row in enumerate(best_rows):
                if i == r:
                    continue
                other_used |= _compute_diffs(row)
            # Try to rebuild row r with backtracking
            new_row = _backtrack_build_row([0], other_used, k, best_max, deadline)
            if new_row is not None:
                new_max = _max_entry(best_rows[:r] + [new_row] + best_rows[r+1:])
                if new_max < best_max:
                    best_max = new_max
                    best_rows = best_rows[:r] + [new_row] + best_rows[r+1:]

    # Phase 4: Final hill-climbing pass
    if best_rows is not None and time.time() < deadline:
        rows, cur_max = _hill_climb(best_rows, k, deadline, rng)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows

    # If we never found a valid solution, fall back to a simple greedy without limit
    if best_rows is None:
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
        best_rows = rows

    return best_rows
