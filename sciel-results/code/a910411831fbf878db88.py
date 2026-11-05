"""Difference triangle set constructor using CSP backtracking with conflict-graph heuristics."""
import time
import itertools
import random

# Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
_DEADLINE = 145.0  # seconds


def _row_diffs(row):
    """Return the set of positive differences for a row (list of increasing ints starting at 0)."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid_row(row):
    """Check that a row is strictly increasing, starts at 0, and all internal diffs are distinct."""
    if not row or row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    diffs = _row_diffs(row)
    return len(diffs) == len(row) * (len(row) - 1) // 2


def _greedy_construct(n_rows, k):
    """Fast greedy baseline: pick smallest next mark whose diffs are unused."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _csp_backtrack(n_rows, k, used_diffs, rows, deadline):
    """
    Backtracking search for the next row given already-used differences.
    Tries to build a row of k+1 increasing ints starting at 0 such that
    all its internal differences are disjoint from used_diffs and from each other.
    Returns the row if found, else None.
    """
    # We build the row position by position.
    # row[0] = 0 always.
    row = [0]
    max_val = 200  # upper bound for values in a row

    def backtrack(pos, last_val):
        if time.time() > deadline:
            return None
        if pos == k + 1:
            return row[:]
        # Try candidate values for row[pos]
        # Heuristic: try smaller values first (tighter packing)
        for val in range(last_val + 1, max_val + 1):
            # Check that val - row[i] for all i < pos are not in used_diffs and not seen in this row
            # Also check that all pairwise diffs within the partial row are distinct
            valid = True
            local_diffs = set()
            for i in range(pos):
                d = val - row[i]
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                local_diffs.add(d)
            if not valid:
                continue
            # Also need to ensure that the diffs among existing row elements are all distinct
            # (they were checked when each was added, so this is fine)
            row.append(val)
            result = backtrack(pos + 1, val)
            if result is not None:
                return result
            row.pop()
        return None

    return backtrack(1, 0)


def _csp_construct(n_rows, k, deadline):
    """
    Build n_rows rows using backtracking, with a cost-based heuristic:
    prefer rows that use 'cheaper' (smaller) differences to leave room for others.
    """
    used = set()
    rows = []
    for r in range(n_rows):
        if time.time() > deadline:
            break
        # Try to find a row with minimal max value (tightest)
        row = _csp_backtrack(n_rows, k, used, rows, deadline)
        if row is None:
            # Fallback: try with a larger max_val by relaxing
            # For now, just use greedy for this row
            x = 0
            row = [0]
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
            continue
        diff_set = _row_diffs(row)
        if diff_set & used:
            # Shouldn't happen, but safety check
            continue
        used |= diff_set
        rows.append(row)
    
    # If we didn't get enough rows, pad with greedy
    while len(rows) < n_rows:
        x = 0
        row = [0]
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows


def _local_search_improve(rows, k, deadline):
    """
    Try to reduce the maximum entry by local search:
    - Identify the row with the largest max value.
    - Try to re-construct that row with a smaller max value,
      keeping other rows' differences fixed.
    """
    if not rows:
        return rows
    
    best_scope = max(max(row) for row in rows)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find row with largest max
        max_idx = max(range(len(rows)), key=lambda i: max(rows[i]))
        current_row = rows[max_idx]
        current_max = max(current_row)
        
        # Compute used diffs from all other rows
        other_used = set()
        for i, row in enumerate(rows):
            if i != max_idx:
                other_used |= _row_diffs(row)
        
        # Try to find a replacement row with smaller max
        for target_max in range(current_max - 1, 0, -1):
            if time.time() > deadline:
                break
            # Try to build a row with max value <= target_max
            found = _try_build_row_with_max(k, target_max, other_used)
            if found is not None:
                rows[max_idx] = found
                new_max = max(found)
                if new_max < best_scope:
                    best_scope = new_max
                    improved = True
                break
    
    return rows


def _try_build_row_with_max(k, target_max, used_diffs):
    """Try to build a row of k+1 increasing ints starting at 0 with max <= target_max."""
    # Use backtracking with the constraint that the last element <= target_max
    row = [0]
    
    def backtrack(pos, last_val):
        if pos == k + 1:
            if last_val <= target_max:
                return row[:]
            return None
        start = last_val + 1
        end = target_max + 1  # inclusive
        for val in range(start, end):
            valid = True
            local_diffs = set()
            for i in range(pos):
                d = val - row[i]
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                local_diffs.add(d)
            if not valid:
                continue
            row.append(val)
            result = backtrack(pos + 1, val)
            if result is not None:
                return result
            row.pop()
        return None
    
    return backtrack(1, 0)


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # First, get a greedy baseline
    rows = _greedy_construct(n_rows, k)
    if time.time() > deadline - 5:
        return rows
    
    # Try CSP-based construction
    try:
        csp_rows = _csp_construct(n_rows, k, deadline - 5)
        if len(csp_rows) == n_rows:
            # Verify validity
            all_diffs = set()
            valid = True
            for row in csp_rows:
                if not _valid_row(row):
                    valid = False
                    break
                rd = _row_diffs(row)
                if rd & all_diffs:
                    valid = False
                    break
                all_diffs |= rd
            if valid:
                rows = csp_rows
    except Exception:
        pass
    
    # Local search to reduce scope
    if time.time() < deadline - 10:
        try:
            rows = _local_search_improve(rows, k, deadline - 10)
        except Exception:
            pass
    
    # Final verification
    all_diffs = set()
    for row in rows:
        if not _valid_row(row):
            break
        rd = _row_diffs(row)
        if rd & all_diffs:
            break
        all_diffs |= rd
    
    return rows
