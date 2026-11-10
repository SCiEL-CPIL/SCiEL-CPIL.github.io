"""Difference triangle set (7,5) – global backtracking with iterative deepening and time guard."""

import time
import random
from itertools import combinations

# Deadline guard: kill after ~110 s to stay safely under the 150 s evaluator limit.
_DEADLINE = 110.0
_START_TIME = None


def _check_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME < _DEADLINE


def _row_diffs(row):
    """Return the set of positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _greedy_row(used, k, max_val):
    """Build one row greedily: start at 0, pick smallest next value whose diffs are unused."""
    row = [0]
    while len(row) <= k:
        x = row[-1] + 1
        while x <= max_val:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                break
            x += 1
        else:
            return None
    return row


def _backtrack_rows(used, rows, n_rows, k, max_val):
    """
    Backtracking search for the remaining rows given current used-diff set and existing rows.
    Returns True on success and fills rows in place; False on failure or timeout.
    """
    if len(rows) == n_rows:
        return True
    if not _check_time():
        return False

    # Try to build the next row with a limited backtracking search.
    # We'll use a recursive helper that picks values one at a time.
    return _backtrack_row(used, rows, n_rows, k, max_val)


def _backtrack_row(used, rows, n_rows, k, max_val, row, pos):
    """
    Recursive helper to build one row of length k+1 starting at 0.
    pos = number of elements already placed in row (row[0] must be 0).
    """
    if len(row) == k + 1:
        rows.append(row)
        if _backtrack_rows(used, rows, n_rows, k, max_val):
            return True
        rows.pop()
        return False

    if not _check_time():
        return False

    start = row[-1] + 1 if row else 1
    for x in range(start, max_val + 1):
        new_diffs = {x - a for a in row}
        if new_diffs & used:
            continue
        used |= new_diffs
        row.append(x)
        if _backtrack_row(used, rows, n_rows, k, max_val, row, pos + 1):
            return True
        row.pop()
        used -= new_diffs
    return False


def _solve_with_maxval(n_rows, k, max_val):
    """
    Try to find n_rows rows of k+1 increasing integers starting at 0,
    all within [0, max_val], with all pairwise differences distinct across rows.
    Returns list of rows on success, None on failure.
    """
    used = set()
    rows = []
    # Build rows one at a time using backtracking.
    # Use a heuristic: try to build each row greedily first, then backtrack if stuck.
    for r in range(n_rows):
        # Greedy attempt
        row = [0]
        greedy_used = set()
        success = True
        while len(row) <= k:
            found = False
            x = row[-1] + 1
            while x <= max_val:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used or new_diffs & greedy_used):
                    row.append(x)
                    greedy_used |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                success = False
                break
        if success:
            used |= greedy_used
            rows.append(row)
        else:
            # Fall back to full backtracking for this row
            row_bt = [0]
            if not _backtrack_row(used, rows, n_rows, k, max_val, row_bt, 0):
                return None
    return rows


def _lower_bound(n_rows, k):
    """Simple counting bound: total distinct diffs needed = n_rows * k*(k+1)/2.
    Max diff <= max_val, so we need max_val*(max_val+1)/2 >= total_diffs."""
    total_diffs = n_rows * k * (k + 1) // 2
    # Solve m*(m+1)/2 >= total_diffs
    import math
    m = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    return m


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all positive differences distinct over all rows.
    """
    global _START_TIME
    _START_TIME = time.time()

    # Start from the counting bound and go up.
    lb = _lower_bound(n_rows, k)
    # Known best for (7,5) is 112; use a reasonable upper bound to start.
    ub = max(lb, 115) if (n_rows, k) == (7, 5) else lb + 20

    best_rows = None
    best_max = None

    # Try increasing max_val from lb upward.
    for max_val in range(lb, ub + 1):
        if not _check_time():
            break
        rows = _solve_with_maxval(n_rows, k, max_val)
        if rows is not None:
            best_rows = rows
            best_max = max_val
            break

    if best_rows is not None:
        return best_rows

    # If iterative deepening didn't find a solution in time, fall back to greedy
    # which always produces a valid (though possibly suboptimal) result.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) <= k:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
