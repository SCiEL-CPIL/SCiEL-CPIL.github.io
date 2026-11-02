"""Difference triangle set (4,7): minimize the largest entry over valid rows."""
import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 130.0  # safety margin under the 150s hard kill
    rng = random.Random(20240517)

    def all_diffs(row):
        out = set()
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - ai)
        return out

    def greedy_row(used, max_mark):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_mark:
                return None
            new = set()
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new.add(d)
            if ok:
                row.append(x)
                used |= new
        return row

    def greedy_solution():
        used, rows = set(), []
        for _ in range(n_rows):
            r = greedy_row(used, 100000)
            if r is None:
                return None
            rows.append(r)
        return rows

    def rebuild(row, used, target, max_mark):
        """Build a row using a fixed set of 'keep' marks; fill the rest greedily."""
        keep = sorted(set(row) - {0})
        row = [0]
        x = 0
        need = k + 1 - len(row)
        for m in keep:
            if m > x:
                x = m
            if len(row) < k + 1:
                new = set()
                ok = True
                for a in row:
                    d = m - a
                    if d in used:
                        ok = False
                        break
                    new.add(d)
                if ok:
                    row.append(m)
                    used |= new
                    need -= 1
        while len(row) < k + 1:
            x += 1
            if x > max_mark:
                return None
            new = set()
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new.add(d)
            if ok:
                row.append(x)
                used |= new
                need -= 1
        return row

    def rebuild_row(rows, idx, used, target, max_mark):
        """Remove row idx, rebuild it to be as small as possible given others."""
        others = set()
        for j, r in enumerate(rows):
            if j != idx:
                others |= all_diffs(r)
        old = rows[idx]
        new = rebuild(old, others, target, max_mark)
        if new is None:
            return None, None
        if max(new) < max(old):
            return new, others
        return None, None

    # --- seed with greedy ---
    rows = greedy_solution()
    if rows is None:
        return [[0] * (k + 1) for _ in range(n_rows)]

    def scope(rs):
        return max(max(r) for r in rs)

    best = [r[:] for r in rows]
    best_scope = scope(best)

    def used_set(rs):
        s = set()
        for r in rs:
            s |= all_diffs(r)
        return s

    cur = [r[:] for r in rows]
    cur_scope = scope(cur)
    T0 = time.time()
    T = max(1.0, best_scope * 0.15)
    T_min = 0.5
    cooldown = 0.0

    while time.time() < deadline:
        T = max(T_min, T * 0.99)
        # pick a row to rebuild; bias toward larger rows
        idx = rng.randrange(n_rows)
        others = set()
        for j, r in enumerate(cur):
            if j != idx:
                others |= all_diffs(r)
        new = rebuild(cur[idx], others, cur_scope - 1, best_scope + 40)
        if new is None:
            continue
        old = cur[idx]
        old_s = max(old)
        new_s = max(new)
        delta = new_s - old_s
        if delta < 0 or (delta <= 0 and rng.random() < 1.0):
            pass
        # accept if it improves global scope, or via SA for small increases
        cand = [r[:] for r in cur]
        cand[idx] = new
        cand_scope = scope(cand)
        dscope = cand_scope - cur_scope
        if dscope < 0:
            cur = cand
            cur_scope = cand_scope
            if cur_scope < best_scope:
                best = [r[:] for r in cur]
                best_scope = cur_scope
                T = max(1.0, best_scope * 0.15)
        elif dscope == 0:
            # neutral move: keep if it shrinks this row (helps future moves)
            if new_s < old_s:
                cur = cand
        else:
            if rng.random() < min(1.0, (-dscope / T)):
                cur = cand
                cur_scope = cand_scope
    return best
# EVOLVE-BLOCK-END
