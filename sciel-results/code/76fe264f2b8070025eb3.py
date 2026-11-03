"""Difference triangle set (4,7): backtracking with bitset difference tracking."""
import time

def construct(n_rows, k):
    deadline = time.time() + 110.0  # safety margin before 120s kill
    total_diffs = n_rows * (k * (k + 1) // 2)  # distinct positive differences needed

    # Try increasing scopes; start near the counting bound
    # counting bound: need total_diffs distinct diffs in [1, m], so m >= total_diffs
    lower_bound = total_diffs
    # known best for (4,7) is 134; give some headroom
    start = max(lower_bound, 120)

    for m in range(start, 200):
        if time.time() > deadline:
            break
        # Try to build n_rows rows each with k+1 marks in [0, m]
        used = [0] * (m + 1)  # used[d] = 1 if difference d is taken
        rows = []
        ok = _build_rows(used, rows, n_rows, k, m, deadline)
        if ok and len(rows) == n_rows:
            return rows

    # Fallback: greedy (always valid)
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


def _build_rows(used, rows, n_rows, k, m, deadline):
    """Try to construct n_rows valid rows. Returns True on success."""
    for _ in range(n_rows):
        if not _build_one_row(used, k, m, deadline):
            return False
    return True


def _build_one_row(used, k, m, deadline):
    """Build one row of k+1 marks (starting at 0) in [0,m] with new differences.
    Uses backtracking. Returns True on success (and appends row to used)."""
    row = [0]
    return _backtrack_row(used, row, 1, k + 1, m, deadline)


def _backtrack_row(used, row, depth, target_len, m, deadline):
    if time.time() > deadline:
        return False
    if depth == target_len:
        return True

    last = row[-1]
    # Try candidate next values from last+1 up to m
    for x in range(last + 1, m + 1):
        # Check all differences x - a for a in row are unused
        new_diffs = [x - a for a in row]
        if any(used[d] for d in new_diffs):
            continue
        # Commit
        for d in new_diffs:
            used[d] = 1
        row.append(x)

        if _backtrack_row(used, row, depth + 1, target_len, m, deadline):
            return True

        # Undo
        row.pop()
        for d in new_diffs:
            used[d] = 0

    return False
