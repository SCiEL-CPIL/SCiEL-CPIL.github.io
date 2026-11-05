"""Time-bounded backtracking solver for the difference triangle set (4,7)."""
import time
import random
import sys

sys.setrecursionlimit(100000)

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin under 150s wall-clock
    need = k + 1  # entries per row

    def row_diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in s:
                    return None
                s.add(d)
        return s

    def build_row(used, max_scope):
        # Build one row of `need` entries starting at 0 within [0, max_scope].
        # used: set of globally used diffs. Returns (row, rowdiffs) or None.
        row = [0]
        all_used = set(used)

        def backtrack(pos, last):
            if time.time() > deadline:
                return False
            if pos == need:
                return True
            for x in range(last + 1, max_scope + 1):
                newdiffs = []
                ok = True
                for a in row:
                    d = x - a
                    if d in all_used:
                        ok = False
                        break
                    if d in newdiffs:
                        ok = False
                        break
                    newdiffs.append(d)
                if not ok:
                    continue
                row.append(x)
                for d in newdiffs:
                    all_used.add(d)
                if backtrack(pos + 1, x):
                    return True
                row.pop()
                for d in newdiffs:
                    all_used.discard(d)
            return False

        if backtrack(1, 0):
            return row, row_diffs(row)
        return None

    def greedy_scope():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < need:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return max(max(r) for r in rows), rows

    g_scope, g_rows = greedy_scope()
    best = g_rows
    best_scope = g_scope

    total_diffs = n_rows * k * (k + 1) // 2
    lo = total_diffs
    hi = g_scope

    def full_search(max_scope):
        used = set()
        rows = []

        def build_row_idx(ri):
            if time.time() > deadline:
                return False
            if ri == n_rows:
                return True
            res = build_row(used, max_scope)
            if res is None:
                return False
            row, rd = res
            used |= rd
            rows.append(row)
            if build_row_idx(ri + 1):
                return True
            rows.pop()
            used -= rd
            return False

        if build_row_idx(0):
            return rows
        return None

    # Binary search on scope with backtracking feasibility.
    attempts = 0
    max_attempts = 60
    while lo < hi and time.time() < deadline - 25 and attempts < max_attempts:
        mid = (lo + hi) // 2
        attempts += 1
        res = full_search(mid)
        if res is not None:
            best = res
            best_scope = max(max(r) for r in res)
            hi = best_scope
        else:
            lo = mid + 1

    # Polish: try to shave scope by rebuilding a random row with a tighter bound.
    if time.time() < deadline - 10 and best is not None:
        rng = random.Random(12345)
        cur = [list(r) for r in best]
        cur_scope = max(max(r) for r in cur)

        def valid(rs):
            used = set()
            for r in rs:
                if r[0] != 0:
                    return False
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        if d in used:
                            return False
                        used.add(d)
            return True

        for _ in range(80):
            if time.time() > deadline:
                break
            ri = rng.randrange(n_rows)
            used = set()
            for idx, r in enumerate(cur):
                if idx == ri:
                    continue
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        used.add(r[j] - r[i])
            new_row_res = build_row(used, cur_scope - 1)
            if new_row_res is not None:
                new_row, _ = new_row_res
                cand = [list(r) for r in cur]
                cand[ri] = new_row
                if valid(cand):
                    ns = max(max(r) for r in cand)
                    if ns < cur_scope:
                        cur = cand
                        cur_scope = ns
        best = cur
        best_scope = cur_scope

    return best
# EVOLVE-BLOCK-END
