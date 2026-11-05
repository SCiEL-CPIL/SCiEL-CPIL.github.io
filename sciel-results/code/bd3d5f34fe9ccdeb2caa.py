import time
import random
from collections import Counter

def construct(n_rows, k):
    """
    Construct a difference triangle set: n_rows rows, each a list of k+1
    strictly increasing integers starting at 0, such that every positive
    difference a_j - a_i (i < j) is distinct across ALL rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s wall clock

    m = k + 1  # elements per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # ---------- fast difference / conflict machinery ----------
    # For a single row, the multiset of its internal differences.
    def row_diffs(row):
        out = []
        for i in range(m):
            ai = row[i]
            for j in range(i + 1, m):
                out.append(row[j] - ai)
        return out

    # Build a global Counter of all differences (across all rows).
    def build_counter(rows):
        c = Counter()
        for r in rows:
            c.update(row_diffs(r))
        return c

    def conflict_count(counter):
        # number of "excess" copies = sum(max(0, cnt-1))
        tot = 0
        for v in counter.values():
            if v > 1:
                tot += v - 1
        return tot

    def is_valid(counter):
        return all(v == 1 for v in counter.values())

    # ---------- initial construction: greedy (valid by construction) ----------
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                # new differences introduced by appending x
                new_diffs = (x - a for a in row)
                if not any(d in used for d in new_diffs):
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    current_rows = greedy_construct()
    counter = build_counter(current_rows)
    cur_conf = conflict_count(counter)
    best_rows = [r[:] for r in current_rows]
    best_scope = max(max(r) for r in best_rows)

    # ---------- helpers for a single-element move ----------
    # Returns (new_row, removed_diffs, added_diffs) for changing
    # row[elem_idx] to new_val (strictly increasing must hold).
    def move_diffs(row, elem_idx, new_val):
        removed = []
        added = []
        # differences involving elem_idx
        for i in range(elem_idx):
            removed.append(row[elem_idx] - row[i])
            added.append(new_val - row[i])
        for j in range(elem_idx + 1, m):
            removed.append(row[j] - row[elem_idx])
            added.append(row[j] - new_val)
        return removed, added

    def apply_move(counter, row, elem_idx, new_val):
        removed, added = move_diffs(row, elem_idx, new_val)
        for d in removed:
            counter[d] -= 1
            if counter[d] == 0:
                del counter[d]
        for d in added:
            counter[d] = counter.get(d, 0) + 1
        row[elem_idx] = new_val

    def undo_move(counter, row, elem_idx, old_val):
        # recompute using current row (which holds new_val) and old_val
        removed, added = move_diffs(row, elem_idx, old_val)  # removed = diffs at new_val, added = diffs at old_val
        # Actually move_diffs returns (removed@current, added@target).
        # We want to go from new_val back to old_val:
        # remove current(new_val) diffs, add old_val diffs.
        for d in removed:
            counter[d] -= 1
            if counter[d] == 0:
                del counter[d]
        for d in added:
            counter[d] = counter.get(d, 0) + 1
        row[elem_idx] = old_val

    # ---------- Phase A: eliminate conflicts (targeted repair + SA) ----------
    def repair_once():
        # Find a difference that appears >1 times; relocate one of its
        # contributing elements to a fresh value.
        # Collect duplicated differences.
        dup = None
        for d, c in counter.items():
            if c > 1:
                dup = d
                break
        if dup is None:
            return False
        # Find a row and element pair (i<j) producing dup.
        for ri in range(n_rows):
            row = current_rows[ri]
            for i in range(m):
                for j in range(i + 1, m):
                    if row[j] - row[i] == dup:
                        # relocate row[j] (the later element) to a new value
                        # that does not create any conflict and keeps order.
                        # Try a few candidate new values.
                        lo = row[j - 1] + 1
                        hi = (row[j + 1] - 1) if j < m - 1 else (best_scope + 40)
                        # candidate set: values that keep row[j] strictly increasing
                        # and do not reuse any used difference.
                        # We'll just try random candidates.
                        for _ in range(60):
                            if lo > hi:
                                break
                            nv = random.randint(lo, hi)
                            if nv == row[j]:
                                continue
                            removed, added = move_diffs(row, j, nv)
                            # simulate
                            ok = True
                            for d in added:
                                c = counter.get(d, 0)
                                if d in removed:
                                    # this diff is being replaced; net effect:
                                    pass
                                if c >= 1 and d not in removed:
                                    ok = False
                                    break
                                if c == 1 and d in removed:
                                    # it was the only copy and we're moving it -> fine
                                    pass
                            if not ok:
                                continue
                            # apply
                            apply_move(counter, row, j, nv)
                            return True
                        break
                break
        return False

    # SA to reduce conflicts.
    if cur_conf > 0:
        temp = 800.0
        cooling = 0.9992
        min_temp = 5.0
        while time.time() < deadline - 25.0 and temp > min_temp:
            # try targeted repair first
            if repair_once():
                cur_conf = conflict_count(counter)
                if cur_conf == 0:
                    break
                continue
            # random single-element move
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, m - 1)
            row = current_rows[ri]
            lo = row[ei - 1] + 1
            hi = (row[ei + 1] - 1) if ei < m - 1 else (best_scope + 30)
            if lo > hi:
                temp *= cooling
                continue
            nv = random.randint(lo, hi)
            if nv == row[ei]:
                temp *= cooling
                continue
            removed, added = move_diffs(row, ei, nv)
            old_conf = cur_conf
            # compute new conflict count incrementally
            # delta in conflict count:
            # For each diff d, conflict contribution = max(0, cnt-1).
            # We'll just apply, measure, and undo if worse.
            apply_move(counter, row, ei, nv)
            new_conf = conflict_count(counter)
            delta = new_conf - old_conf
            if delta < 0 or (delta == 0 and random.random() < 0.3) or (delta > 0 and random.random() < pow(2.718281828459045, -delta / temp)):
                cur_conf = new_conf
                # update best if valid
                if cur_conf == 0:
                    s = max(max(r) for r in current_rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [r[:] for r in current_rows]
            else:
                undo_move(counter, row, ei, nv)
            temp *= cooling

    # ---------- Phase B: minimize scope ----------
    # Ensure we start from a valid state for scope minimization.
    if cur_conf != 0:
        # fall back: if not valid, try a fresh greedy + quick repair loop
        pass

    # Rebuild from best valid rows if we have one, else from current.
    if is_valid(build_counter(best_rows)):
        work_rows = [r[:] for r in best_rows]
        work_counter = build_counter(work_rows)
    else:
        work_rows = [r[:] for r in current_rows]
        work_counter = build_counter(work_rows)

    # Scope minimization via SA + greedy shrink.
    work_conf = conflict_count(work_counter)
    work_scope = max(max(r) for r in work_rows)

    # Greedy shrink: try to reduce the maximum element by lowering it.
    def greedy_shrink(rows, counter):
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            # find the row with the maximum element
            max_val = -1
            max_pos = None
            for ri in range(n_rows):
                for ei in range(m):
                    if rows[ri][ei] > max_val:
                        max_val = rows[ri][ei]
                        max_pos = (ri, ei)
            if max_pos is None:
                break
            ri, ei = max_pos
            row = rows[ri]
            lo = row[ei - 1] + 1 if ei > 0 else 1
            hi = max_val - 1
            if hi < lo:
                break
            # try to lower to some value in [lo, hi]
            for _ in range(40):
                nv = random.randint(lo, hi)
                if nv == row[ei]:
                    continue
                removed, added = move_diffs(row, ei, nv)
                # check no conflict introduced
                ok = True
                for d in added:
                    if d in removed:
                        continue
                    if counter.get(d, 0) >= 1:
                        ok = False
                        break
                if not ok:
                    continue
                apply_move(counter, row, ei, nv)
                improved = True
                break
        return rows, counter

    work_rows, work_counter = greedy_shrink(work_rows, work_counter)
    work_scope = max(max(r) for r in work_rows)
    if work_scope < best_scope:
        best_scope = work_scope
        best_rows = [r[:] for r in work_rows]

    # SA on scope (only accept moves that keep conflicts at 0).
    temp = 150.0
    cooling = 0.9985
    min_temp = 1.0
    while time.time() < deadline - 8.0 and temp > min_temp:
        ri = random.randint(0, n_rows - 1)
        ei = random.randint(1, m - 1)
        row = work_rows[ri]
        lo = row[ei - 1] + 1
        hi = (row[ei + 1] - 1) if ei < m - 1 else (best_scope + 5)
        if lo > hi:
            temp *= cooling
            continue
        # bias toward smaller values to shrink scope
        nv = random.randint(lo, hi)
        if nv == row[ei]:
            temp *= cooling
            continue
        removed, added = move_diffs(row, ei, nv)
        # must not introduce conflicts
        ok = True
        for d in added:
            if d in removed:
                continue
            if work_counter.get(d, 0) >= 1:
                ok = False
                break
        if not ok:
            temp *= cooling
            continue
        apply_move(work_counter, row, ei, nv)
        new_scope = max(max(r) for r in work_rows)
        delta = new_scope - work_scope
        if delta < 0 or random.random() < pow(2.718281828459045, -delta / temp):
            work_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in work_rows]
        else:
            undo_move(work_counter, row, ei, nv)
        temp *= cooling

    # ---------- final validation & return ----------
    final_counter = build_counter(best_rows)
    if not is_valid(final_counter):
        # try to return a valid one from work/current if valid
        if is_valid(build_counter(work_rows)):
            return work_rows
        if is_valid(build_counter(current_rows)):
            return current_rows
    return best_rows
