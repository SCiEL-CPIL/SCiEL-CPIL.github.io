"""Difference triangle set (n_rows, k): rows of k+1 increasing ints from 0,
all intra-row positive differences distinct globally; minimize the largest entry."""
import time

_START = time.time()
_DEADLINE = 150.0 - 5.0  # safety margin before the 150s kill

def _deadline():
    return time.time() >= _DEADLINE

def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s

def _greedy_row(k, used):
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row

def _try_move_last(rows, used, deadline):
    """Try to reduce the last mark of each row. Returns improved flag."""
    improved = False
    n = len(rows)
    for i in range(n):
        if time.time() > deadline:
            break
        row = rows[i]
        old_last = row[-1]
        # differences involving the last element
        old_new = {old_last - a for a in row[:-1]}
        used -= old_new
        # try smaller values
        for v in range(old_last - 1, row[-2], -1):
            if time.time() > deadline:
                break
            cand_new = {v - a for a in row[:-1]}
            if not (cand_new & used):
                row[-1] = v
                used |= cand_new
                improved = True
                break
        else:
            # no improvement; restore
            used |= old_new
    return improved

def construct(n_rows, k):
    # --- Phase 1: greedy rulers ---
    used, rows = set(), []
    for _ in range(n_rows):
        row = _greedy_row(k, used)
        rows.append(row)

    # --- Phase 2: local search to shrink rows ---
    best = max(max(r) for r in rows)
    rows = [list(r) for r in rows]
    row_diffs = [_row_diffs(r) for r in rows]
    all_used = set()
    for s in row_diffs:
        all_used |= s
    used = set(all_used)

    def try_move(r, i, j):
        nonlocal rows, row_diffs, used, all_used, best
        row = rows[r]
        old = row[j]
        cur = {row[t] - row[i] for t in range(i + 1, len(row))}
        lo = row[i] + 1
        for v in range(lo, old):
            if v in used:
                continue
            nd = {v - row[t] for t in range(i + 1, j)}
            if not (nd & all_used):
                row[j] = v
                row_diffs[r] = (row_diffs[r] - cur) | nd
                used = (used - cur) | nd
                all_used = (all_used - cur) | nd
                return v
        return None

    while not _deadline():
        # First pass: try to shrink the last element of each row (fast)
        improved_last = _try_move_last(rows, used, _DEADLINE)
        
        # Second pass: full local search on interior elements
        moved = False
        for r in range(n_rows - 1, -1, -1):
            if _deadline():
                break
            row = rows[r]
            j = len(row) - 1
            while j >= 1 and not _deadline():
                i = j - 1
                while i >= 0 and not _deadline():
                    v = try_move(r, i, j)
                    if v is not None:
                        last = row[-1]
                        if last < best:
                            best = last
                            moved = True
                        break
                    i -= 1
                if v is not None:
                    break
                j -= 1
            if _deadline():
                break
        
        if not moved and not improved_last:
            break

    return rows
