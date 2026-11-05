import time
import random


def construct(n_rows, k):
    random.seed(1234567)
    start = time.time()
    DEADLINE = 145.0  # 150s wall-clock limit, keep a safety margin

    n = k + 1                      # entries per row
    diffs_per_row = n * (n - 1) // 2
    total_diffs = n_rows * diffs_per_row
    S_min = total_diffs            # counting lower bound on the scope

    def now():
        return time.time()

    def time_left():
        return DEADLINE - now()

    # ---------- fast greedy: always yields a valid (maybe large) solution ----------
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < n:
                v = row[-1] + 1
                while True:
                    ok = True
                    for q in range(len(row)):
                        if (v - row[q]) in used:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for q in range(len(row) - 1):
                    used.add(v - row[q])
            rows.append(row)
        return rows

    best_rows = greedy_solution()
    best_S = max(max(r) for r in best_rows)

    # ---------- validation ----------
    def validate(rows):
        all_d = set()
        for r in rows:
            if len(r) != n or r[0] != 0:
                return False
            for i in range(1, n):
                if r[i] <= r[i - 1]:
                    return False
            for i in range(n):
                for j in range(i + 1, n):
                    d = r[j] - r[i]
                    if d in all_d:
                        return False
                    all_d.add(d)
        return True

    # ---------- CSP backtracking with symmetry breaking + fast diff flags ----------
    def solve_for_S(S, time_budget, rng):
        used_flag = [False] * (S + 1)
        rows = [[] for _ in range(n_rows)]
        t0 = now()
        nodes = 0

        def bt(r, p, prev_max):
            nonlocal nodes
            nodes += 1
            if (nodes & 1023) == 0 and now() - t0 > time_budget:
                return False
            if r == n_rows:
                return True
            row = rows[r]
            if p == n:
                if row[-1] < prev_max:      # symmetry: non-decreasing maxima
                    return False
                return bt(r + 1, 0, row[-1])
            if p == 0:
                row.append(0)
                res = bt(r, 1, prev_max)
                row.pop()
                return res
            lo = row[p - 1] + 1
            rem = n - 1 - p
            hi = S - rem
            if hi > S:
                hi = S
            if hi < lo:
                return False
            if rng.random() < 0.5:
                vals = list(range(lo, hi + 1))
                rng.shuffle(vals)
            else:
                vals = range(lo, hi + 1)
            for v in vals:
                ok = True
                for q in range(p):
                    if used_flag[v - row[q]]:
                        ok = False
                        break
                if not ok:
                    continue
                row.append(v)
                added = [v - row[q] for q in range(p)]
                for d in added:
                    used_flag[d] = True
                if bt(r, p + 1, prev_max):
                    return True
                for d in added:
                    used_flag[d] = False
                row.pop()
            return False

        if bt(0, 0, -1):
            return [list(r) for r in rows]
        return None

    def try_scope(S, per_S_budget, seed_base):
        """Run randomized restarts of the CSP for cap S; return solution or None."""
        t0s = now()
        restart = 0
        while now() - t0s < per_S_budget and time_left() > 8:
            rng = random.Random(seed_base * S + 7 * restart + 13)
            res = solve_for_S(S, time_left() - 8, rng)
            if res is not None:
                return res
            restart += 1
        return None

    # ---------- iterative deepening on the scope (ascending) ----------
    S = S_min
    while S <= best_S and time_left() > 10:
        if S < 110:
            per_S = 8.0
        elif S < 115:
            per_S = 5.0
        else:
            per_S = 3.0
        per_S = min(per_S, time_left() - 10)
        found = try_scope(S, per_S, 1000003)
        if found is not None and validate(found):
            sc = max(max(r) for r in found)
            if sc < best_S:
                best_S = sc
                best_rows = [list(r) for r in found]
            break
        S += 1

    # ---------- refinement: probe just below the achieved scope ----------
    S = best_S - 1
    while S >= S_min and time_left() > 6:
        per_S = min(4.0, time_left() - 6)
        found = try_scope(S, per_S, 2000003)
        if found is not None and validate(found):
            sc = max(max(r) for r in found)
            if sc < best_S:
                best_S = sc
                best_rows = [list(r) for r in found]
        S -= 1

    # ---------- light local-search polish (time-boxed) ----------
    def local_search(rows, time_budget):
        rows = [list(r) for r in rows]
        t0 = now()

        def row_diffs(row):
            s = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
            return s

        improved = True
        while improved and now() - t0 < time_budget:
            improved = False
            order = list(range(n_rows))
            random.shuffle(order)
            for i in order:
                if now() - t0 >= time_budget:
                    break
                row = rows[i]
                od = set()
                for j in range(n_rows):
                    if j != i:
                        od |= row_diffs(rows[j])
                for p in range(len(row) - 1, 0, -1):
                    cur = row[p]
                    lo = row[p - 1] + 1
                    hi = cur - 1
                    if hi < lo:
                        continue
                    for v in range(lo, hi + 1):
                        if p < len(row) - 1 and v >= row[p + 1]:
                            continue
                        cand = list(row)
                        cand[p] = v
                        cd = row_diffs(cand)
                        if len(cd) != diffs_per_row:
                            continue
                        if cd & od:
                            continue
                        rows[i] = cand
                        improved = True
                        break
                    if improved:
                        break
        return rows

    if time_left() > 8:
        ls = local_search(best_rows, time_left() - 6)
        if validate(ls):
            sc = max(max(r) for r in ls)
            if sc < best_S:
                best_S = sc
                best_rows = ls

    # ---------- final safety net ----------
    if not validate(best_rows):
        best_rows = greedy_solution()

    return best_rows
