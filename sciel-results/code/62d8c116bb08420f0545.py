"""Difference triangle set (4,7): bitmask fast construction + randomized local search."""
import time
import random


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s kill

    # Fast bitmask validity: difference d is "used" iff (used_mask >> d) & 1.
    # A candidate x for row r is valid iff no (x - a) for a in row r is in used_mask.
    def row_valid_mask(row, x, used_mask):
        # Build a bitmask of all differences x - a for a in row; check overlap with used_mask.
        m = 0
        for a in row:
            m |= 1 << (x - a)
        return (m & used_mask) == 0, m

    def next_valid(row, used_mask, lo, hi):
        # smallest x in [lo, hi] valid for this row; return (x, diff_mask) or (None, 0)
        for x in range(lo, hi + 1):
            ok, dm = row_valid_mask(row, x, used_mask)
            if ok:
                return x, dm
        return None, 0

    # ---- Construction: build rows one at a time in a random order, smallest-first.
    def construct_rows(order, used_mask, rows, max_val):
        # order: list of row indices to fill
        for i in order:
            row = [0]
            while len(row) < k + 1:
                x, dm = next_valid(row, used_mask, row[-1] + 1, max_val)
                if x is None:
                    return False
                row.append(x)
                used_mask |= dm
            rows[i] = row
        return True

    def greedy_random(order, max_val):
        rows = [None] * n_rows
        if construct_rows(order, 0, rows, max_val):
            return rows
        return None

    # ---- Baseline: deterministic greedy (row order 0..n_rows-1) with a generous bound.
    best_rows = greedy_random(list(range(n_rows)), 1000)
    best_max = max(max(r) for r in best_rows) if best_rows else 1000

    # ---- Phase 1: randomized restarts of row-order construction.
    # Each restart fills rows in a random permutation; keep the best max.
    rng = random.Random(12345)
    for _ in range(20000):
        if time.time() > deadline:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = greedy_random(order, best_max)
        if rows:
            m = max(max(r) for r in rows)
            if m < best_max:
                best_max = m
                best_rows = rows

    # ---- Phase 2: local search. Rebuild one row at a time with a tighter max bound,
    #    trying to lower the global max. Also try swapping the last element of a row.
    def rebuild_row(rows, idx, max_val):
        # remove row idx, rebuild it given the other rows' differences.
        saved = rows[idx]
        rows[idx] = None
        # recompute used_mask from all other rows
        used_mask = 0
        for j in range(n_rows):
            if j == idx:
                continue
            r = rows[j]
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    used_mask |= 1 << (r[b] - r[a])
        new_row = [0]
        ok = True
        while len(new_row) < k + 1:
            x, dm = next_valid(new_row, used_mask, new_row[-1] + 1, max_val)
            if x is None:
                ok = False
                break
            new_row.append(x)
            used_mask |= dm
        if not ok:
            rows[idx] = saved
            return False
        rows[idx] = new_row
        return True

    # Try to reduce best_max: for each target max below current, try to fit all rows.
    # We do this by greedily rebuilding rows under a tighter bound.
    improved = True
    while improved:
        improved = False
        for target in range(best_max - 1, max(112, best_max - 15), -1):
            if time.time() > deadline:
                return best_rows
            # try to rebuild each row under 'target' bound (keep others fixed)
            work = [list(r) for r in best_rows]
            success = True
            # order of rebuilding: rows with largest last element first (most constrained)
            order = sorted(range(n_rows), key=lambda i: work[i][-1], reverse=True)
            for i in order:
                if not rebuild_row(work, i, target):
                    success = False
                    break
            if success:
                m = max(max(r) for r in work)
                if m <= target:
                    best_rows = work
                    best_max = m
                    improved = True
                    break

    return best_rows
