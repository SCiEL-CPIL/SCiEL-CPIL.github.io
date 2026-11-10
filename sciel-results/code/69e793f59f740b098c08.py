import time

def construct(n_rows, k):
    start = time.time()
    deadline = start + 135  # safety margin before 150s wall-clock

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None):
        """Smallest-next-mark row avoiding used_set. Returns None if cap too tight."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    def dfs_row(row, used_set, cap, row_deadline):
        """Find a row with last element <= cap avoiding used_set. Returns row or None."""
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)  # each remaining step adds at least 1
        if upper < last + 1:
            return None
        for x in range(last + 1, upper + 1):
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline)
            if res is not None:
                return res
        return None

    # ---- Phase 1: greedy baseline (always valid) ----
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        rows.append(row)
        used |= row_diffs(row)

    # ---- Phase 2: per-row DFS refinement during construction ----
    # Re-do each row with DFS using the current used set, trying to shrink it.
    # This is inspired by the "DFS per row" approach but applied iteratively.
    
    # ---- Phase 3: bounded iterative improvement (replace rows with smaller ones) ----
    # A replaced row frees its old differences, which can let a *smaller* row fit.
    # We only ever shrink the max entry, so the result stays valid and improves.
    order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for r in order:
            if time.time() > deadline - 5:
                break
            cur_last = rows[r][-1]
            # Try to fit a strictly smaller row using freed slots.
            best = None
            # Start from cur_last - 1 and go down, but limit the search range
            # to avoid excessive DFS calls. We'll try a few caps.
            for cap in range(cur_last - 1, max(cur_last - 30, 0), -1):
                if time.time() > deadline - 5:
                    break
                used_without = set(used)
                used_without.difference_update(row_diffs(rows[r]))
                row_deadline = min(deadline - 5, time.time() + 2)
                cand = dfs_row([0], used_without, cap, row_deadline)
                if cand is not None:
                    best = cand
                    break
            if best is not None and best[-1] < cur_last:
                used.difference_update(row_diffs(rows[r]))
                rows[r] = best
                used |= row_diffs(best)
                improved = True

    return rows
