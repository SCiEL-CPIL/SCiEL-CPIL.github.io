import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin before 150s kill

    n = k + 1
    diffs_per_row = n * k // 2
    total_diffs = n_rows * diffs_per_row

    # ---- Fast difference helpers (bitmask over a 64-bit word) ----
    def row_mask(row):
        m = 0
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= 1 << (row[j] - ri)
        return m

    # ---- Initial valid solution via greedy (always valid, may be large) ----
    def greedy_construct():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < n:
                x = row[-1] + 1
                while True:
                    ok = True
                    for y in row:
                        if (1 << (x - y)) & used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for y in row[:-1]:
                            used |= 1 << (x - y)
                        break
                    x += 1
            rows.append(row)
        return rows

    best = greedy_construct()
    best_scope = max(max(r) for r in best)
    best_masks = [row_mask(r) for r in best]
    best_union = 0
    for mm in best_masks:
        best_union |= mm

    # ---- Local search over a scope-decreasing objective ----
    def local_search(rows, masks, union, scope, t_end):
        cur_rows = [r[:] for r in rows]
        cur_masks = [m for m in masks]
        cur_union = union
        cur_scope = scope
        loc_best = cur_scope
        loc_best_rows = [r[:] for r in cur_rows]

        while time.time() < t_end:
            row_idx = random.randrange(n_rows)
            pos = random.randrange(1, n)
            row = cur_rows[row_idx]
            old = row[pos]
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1 if pos < n - 1 else cur_scope
            if hi < lo:
                continue

            old_mask = cur_masks[row_idx]
            base = cur_union & ~old_mask
            improved = False
            # Prefer the largest value that keeps scope from growing (shrinks scope if it is the max)
            for v in range(hi, lo - 1, -1):
                # quick scope guard: don't grow the max
                if v > cur_scope and pos == n - 1:
                    continue
                row[pos] = v
                m = row_mask(row)
                if (m & base) != 0:
                    continue
                new_scope = cur_scope
                if pos == n - 1 and v > new_scope:
                    new_scope = v
                # accept if scope strictly decreases (or tie, to wander)
                if new_scope < cur_scope or (new_scope == cur_scope and random.random() < 0.15):
                    cur_masks[row_idx] = m
                    cur_union = base | m
                    cur_scope = new_scope
                    improved = True
                    if cur_scope < loc_best:
                        loc_best = cur_scope
                        loc_best_rows = [r[:] for r in cur_rows]
                    break
            if not improved:
                row[pos] = old  # restore (unchanged)
        return loc_best, loc_best_rows, cur_rows, cur_masks, cur_union, cur_scope

    # ---- Diversified random restarts ----
    def random_valid_rows():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < n:
                x = row[-1] + random.randrange(1, 4)
                while True:
                    ok = True
                    for y in row:
                        if (1 << (x - y)) & used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for y in row[:-1]:
                            used |= 1 << (x - y)
                        break
                    x += 1
            rows.append(row)
        return rows

    while time.time() < deadline - 3.0:
        if time.time() < deadline - 6.0 and random.random() < 0.35:
            start_rows = random_valid_rows()
        else:
            start_rows = [r[:] for r in best]
        masks = [row_mask(r) for r in start_rows]
        union = 0
        for mm in masks:
            union |= mm
        scope = max(max(r) for r in start_rows)

        t_end = min(time.time() + 4.0, deadline - 3.0)
        lb, lbr, _, _, _, _ = local_search(start_rows, masks, union, scope, t_end)
        if lb < best_scope:
            best_scope = lb
            best = [r[:] for r in lbr]
            best_masks = [row_mask(r) for r in best]
            best_union = 0
            for mm in best_masks:
                best_union |= mm

    # Final safety: ensure validity of returned best (it is, by construction)
    return best
