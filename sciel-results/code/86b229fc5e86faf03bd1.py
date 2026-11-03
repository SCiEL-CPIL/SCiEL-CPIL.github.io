import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 135.0  # safety margin before 150s kill
    row_len = k + 1

    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def greedy_row(used_mask, max_scope=200):
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            if x > max_scope:
                return None
            new_mask = 0
            for a in row:
                new_mask |= (1 << (x - a))
            if (new_mask & used_mask) == 0:
                row.append(x)
                used_mask |= new_mask
        return row

    def find_row_dfs(used_mask, best_max_init, S_cap, node_budget, dl):
        best = [None]
        best_max = [best_max_init]
        nodes = [0]

        def dfs(row, rmask, last):
            if time.time() > dl or nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if len(row) == row_len:
                m = row[-1]
                if m < best_max[0]:
                    best_max[0] = m
                    best[0] = row[:]
                return
            hi = min(best_max[0] - 1, S_cap)
            lo = last + 1
            if lo > hi:
                return
            for x in range(lo, hi + 1):
                new_mask = 0
                for a in row:
                    new_mask |= (1 << (x - a))
                if (new_mask & used_mask) or (new_mask & rmask):
                    continue
                row.append(x)
                dfs(row, rmask | new_mask, x)
                row.pop()
                if nodes[0] >= node_budget or time.time() > dl:
                    return

        dfs([0], 0, 0)
        if best[0] is None:
            return None
        return best[0], row_mask(best[0])

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def build_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                ri = r[i]
                for j in range(i + 1, len(r)):
                    d = r[j] - ri
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def remove_row_diffs(dc, row):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                dc[d] -= 1
                if dc[d] == 0:
                    del dc[d]

    def add_row_diffs(dc, row):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                dc[d] = dc.get(d, 0) + 1

    def row_rebuild_greedy(used_mask, max_scope, rng, attempts=5):
        for attempt in range(attempts):
            row = [0]
            x = 0
            local_mask = 0
            for j in range(1, row_len):
                found = False
                start = x + 1
                if attempt > 0 and rng.random() < 0.4:
                    start = x + rng.randint(1, 4)
                for t in range(start, max_scope + 1):
                    new_mask = 0
                    for a in row:
                        new_mask |= (1 << (t - a))
                    if (new_mask & used_mask) or (new_mask & local_mask):
                        continue
                    row.append(t)
                    local_mask |= new_mask
                    x = t
                    found = True
                    break
                if not found:
                    return None
            return row, row_mask(row)
        return None

    def sa_optimize(rows, k, iters, rng, deadline):
        n = len(rows)
        cur_rows = [list(r) for r in rows]
        cur_scope = scope_of(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_scope = cur_scope

        diff_counts = build_diff_counts(cur_rows)

        def get_cost(rows, dc):
            scope = scope_of(rows)
            collisions = sum(max(0, c - 1) for c in dc.values())
            return scope + 1000 * collisions

        cur_cost = get_cost(cur_rows, diff_counts)
        best_cost = cur_cost

        T = 8.0
        cooling = 0.9995
        min_T = 0.1

        # Precompute used masks for each row exclusion
        def compute_used_without(i):
            um = 0
            for ri in range(n):
                if ri != i:
                    um |= row_mask(cur_rows[ri])
            return um

        for it in range(iters):
            if time.time() > deadline:
                break

            r = rng.random()

            if r < 0.30:
                # Row rebuild
                i = rng.randint(0, n - 1)
                old_row = list(cur_rows[i])
                used_without = compute_used_without(i)

                res = row_rebuild_greedy(used_without, cur_scope + 5, rng)
                if res is None:
                    continue
                new_row, new_mask = res

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                new_cost = get_cost(cur_rows, diff_counts)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = scope_of(cur_rows)
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row

            elif r < 0.55:
                # Scope compression: reduce max element via DFS
                max_val = cur_scope
                max_row_idx = -1
                for ri in range(n):
                    if cur_rows[ri][-1] == max_val:
                        max_row_idx = ri
                        break
                if max_row_idx < 0:
                    continue

                i = max_row_idx
                old_row = list(cur_rows[i])
                used_without = compute_used_without(i)

                res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, 30000, min(deadline, time.time() + 1.5))
                if res is not None:
                    new_row, new_mask = res
                    remove_row_diffs(diff_counts, old_row)
                    add_row_diffs(diff_counts, new_row)
                    cur_rows[i] = new_row
                    new_cost = get_cost(cur_rows, diff_counts)

                    if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                        cur_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_scope = scope_of(cur_rows)
                            best_rows = [list(r) for r in cur_rows]
                    else:
                        remove_row_diffs(diff_counts, new_row)
                        add_row_diffs(diff_counts, old_row)
                        cur_rows[i] = old_row

            else:
                # Single element move
                i = rng.randint(0, n - 1)
                j = rng.randint(1, row_len - 1)
                old_row = list(cur_rows[i])
                used_without = compute_used_without(i)

                lo = old_row[j - 1] + 1
                hi = old_row[j + 1] - 1 if j < row_len - 1 else cur_scope + 3
                if lo >= hi:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_row[j]:
                    continue

                new_row = list(old_row)
                new_row[j] = new_val
                if new_row[j] <= new_row[j - 1] or (j < row_len - 1 and new_row[j] >= new_row[j + 1]):
                    continue

                new_mask = row_mask(new_row)
                if new_mask & used_without:
                    continue

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                new_cost = get_cost(cur_rows, diff_counts)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = scope_of(cur_rows)
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row

            T = max(min_T, T * cooling)

        return best_scope, best_rows

    def run(seed, dl):
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        # Try to build rows with DFS first, falling back to greedy
        for _ in range(n_rows):
            if time.time() > dl:
                break
            res = find_row_dfs(used_mask, 200, 200, 80000, dl)
            if res is None:
                row = greedy_row(used_mask, 200)
                if row is None:
                    row = greedy_row(used_mask, 300)
                if row is None:
                    break
                used_mask |= row_mask(row)
            else:
                row, rm = res
                used_mask |= rm
            rows.append(row)

        while len(rows) < n_rows:
            row = greedy_row(used_mask, 300)
            if row is None:
                break
            used_mask |= row_mask(row)
            rows.append(row)

        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)

        # SA refinement
        if time.time() < dl - 3:
            sa_scope, sa_rows = sa_optimize(rows, k, 8000, rng, dl - 1)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]

        return best_scope, best_rows

    try:
        overall = None
        seed = 0
        first = True
        while time.time() < DEADLINE - 5:
            if first:
                dl = min(DEADLINE - 3, time.time() + 55)
                first = False
            else:
                dl = min(DEADLINE - 3, time.time() + 14)
            sc, rows = run(seed, dl)
            if overall is None or sc < overall[0]:
                overall = (sc, rows)
            seed += 1
        if overall is not None:
            return overall[1]
    except Exception:
        pass

    # Fallback: pure greedy
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used_mask, 500)
        if row is None:
            break
        used_mask |= row_mask(row)
        rows.append(row)
    while len(rows) < n_rows:
        row = greedy_row(used_mask, 500)
        if row is None:
            break
        used_mask |= row_mask(row)
        rows.append(row)
    return rows
