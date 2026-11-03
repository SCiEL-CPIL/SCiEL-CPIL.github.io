import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0
    row_len = k + 1

    # --- Bitmask helpers (fast for large scopes) ---
    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def greedy_row(used_mask):
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            new_mask = 0
            for a in row:
                new_mask |= (1 << (x - a))
            if (new_mask & used_mask) == 0:
                row.append(x)
                used_mask |= new_mask
        return row

    # --- Difference-space helpers (compact representation) ---
    def row_from_diffs(diffs):
        row = [0]
        for d in diffs:
            row.append(row[-1] + d)
        return row

    def diffs_from_row(row):
        return [row[i + 1] - row[i] for i in range(len(row) - 1)]

    def pairwise_diffs(row):
        L = len(row)
        diffs = []
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                diffs.append(row[j] - ri)
        return diffs

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def sum_entries(rows):
        return sum(sum(r) for r in rows)

    def build_diff_counts(rows):
        counts = Counter()
        for r in rows:
            for d in pairwise_diffs(r):
                counts[d] = counts.get(d, 0) + 1
        return counts

    def remove_row_diffs(dc, row):
        for d in pairwise_diffs(row):
            dc[d] -= 1
            if dc[d] == 0:
                del dc[d]

    def add_row_diffs(dc, row):
        for d in pairwise_diffs(row):
            dc[d] = dc.get(d, 0) + 1

    # --- Targeted DFS to find a row with max < target_max ---
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

    # --- SA in difference-space with sum-of-entries secondary cost ---
    def sa_optimize(rows, iters, rng, deadline):
        n = len(rows)
        cur_rows = [list(r) for r in rows]
        cur_diffs = [diffs_from_row(r) for r in cur_rows]
        cur_scope = scope_of(cur_rows)
        cur_sum = sum_entries(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_diffs = [d[:] for d in cur_diffs]
        best_scope = cur_scope
        best_sum = cur_sum

        diff_counts = build_diff_counts(cur_rows)
        row_masks = [row_mask(r) for r in cur_rows]

        def compute_used_without(i):
            u = 0
            for ri in range(n):
                if ri != i:
                    u |= row_masks[ri]
            return u

        def get_cost(rows, dc, s):
            collisions = sum(max(0, c - 1) for c in dc.values())
            scope = scope_of(rows)
            # Secondary cost: sum of entries (guides packing density)
            # Weight tuned so that reducing scope is primary, sum is secondary
            return scope + 0.01 * s + 1000 * collisions

        cur_cost = get_cost(cur_rows, diff_counts, cur_sum)
        best_cost = cur_cost

        T = 5.0
        cooling = 0.9997
        min_T = 0.05

        for it in range(iters):
            if time.time() > deadline:
                break

            r = rng.random()

            if r < 0.30:
                # Row rebuild in difference-space
                i = rng.randint(0, n - 1)
                old_row = list(cur_rows[i])
                old_diffs = cur_diffs[i][:]
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                # Try greedy rebuild with randomization
                new_row = None
                for attempt in range(3):
                    row = [0]
                    x = 0
                    local_mask = 0
                    success = True
                    for j in range(1, row_len):
                        found = False
                        start = x + 1
                        if attempt > 0 and rng.random() < 0.3:
                            start = x + rng.randint(1, 3)
                        for t in range(start, cur_scope + 3):
                            new_mask = 0
                            for a in row:
                                new_mask |= (1 << (t - a))
                            if (new_mask & used_without) or (new_mask & local_mask):
                                continue
                            row.append(t)
                            local_mask |= new_mask
                            x = t
                            found = True
                            break
                        if not found:
                            success = False
                            break
                    if success:
                        new_row = row
                        break

                if new_row is None:
                    continue

                new_diffs = diffs_from_row(new_row)
                new_mask = row_mask(new_row)

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                cur_diffs[i] = new_diffs
                row_masks[i] = new_mask
                new_sum = sum_entries(cur_rows)
                new_cost = get_cost(cur_rows, diff_counts, new_sum)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    cur_sum = new_sum
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_sum = cur_sum
                        best_rows = [list(r) for r in cur_rows]
                        best_diffs = [d[:] for d in cur_diffs]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    cur_diffs[i] = old_diffs
                    row_masks[i] = old_mask

            elif r < 0.50:
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
                old_diffs = cur_diffs[i][:]
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, 50000, min(deadline, time.time() + 1.5))
                if res is not None:
                    new_row, new_mask = res
                    new_diffs = diffs_from_row(new_row)
                    remove_row_diffs(diff_counts, old_row)
                    add_row_diffs(diff_counts, new_row)
                    cur_rows[i] = new_row
                    cur_diffs[i] = new_diffs
                    row_masks[i] = new_mask
                    new_sum = sum_entries(cur_rows)
                    new_cost = get_cost(cur_rows, diff_counts, new_sum)

                    if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                        cur_cost = new_cost
                        cur_scope = scope_of(cur_rows)
                        cur_sum = new_sum
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_scope = cur_scope
                            best_sum = cur_sum
                            best_rows = [list(r) for r in cur_rows]
                            best_diffs = [d[:] for d in cur_diffs]
                    else:
                        remove_row_diffs(diff_counts, new_row)
                        add_row_diffs(diff_counts, old_row)
                        cur_rows[i] = old_row
                        cur_diffs[i] = old_diffs
                        row_masks[i] = old_mask

            elif r < 0.75:
                # Single difference mutation (difference-space)
                i = rng.randint(0, n - 1)
                pos = rng.randint(0, k - 1)
                old_row = list(cur_rows[i])
                old_diffs = cur_diffs[i][:]
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                # Bias toward smaller values to reduce scope
                if rng.random() < 0.6:
                    delta = rng.randint(-2, 1)
                else:
                    delta = rng.randint(-3, 3)
                new_diff = max(1, old_diffs[pos] + delta)
                if new_diff == old_diffs[pos]:
                    continue

                new_diffs = old_diffs[:]
                new_diffs[pos] = new_diff
                new_row = row_from_diffs(new_diffs)

                new_mask = row_mask(new_row)
                if new_mask & used_without:
                    continue

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                cur_diffs[i] = new_diffs
                row_masks[i] = new_mask
                new_sum = sum_entries(cur_rows)
                new_cost = get_cost(cur_rows, diff_counts, new_sum)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    cur_sum = new_sum
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_sum = cur_sum
                        best_rows = [list(r) for r in cur_rows]
                        best_diffs = [d[:] for d in cur_diffs]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    cur_diffs[i] = old_diffs
                    row_masks[i] = old_mask

            else:
                # Swap two differences within a row
                i = rng.randint(0, n - 1)
                p1 = rng.randint(0, k - 1)
                p2 = rng.randint(0, k - 1)
                if p1 == p2:
                    continue

                old_row = list(cur_rows[i])
                old_diffs = cur_diffs[i][:]
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                new_diffs = old_diffs[:]
                new_diffs[p1], new_diffs[p2] = old_diffs[p2], old_diffs[p1]
                if new_diffs == old_diffs:
                    continue

                new_row = row_from_diffs(new_diffs)
                new_mask = row_mask(new_row)
                if new_mask & used_without:
                    continue

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                cur_diffs[i] = new_diffs
                row_masks[i] = new_mask
                new_sum = sum_entries(cur_rows)
                new_cost = get_cost(cur_rows, diff_counts, new_sum)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    cur_sum = new_sum
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_sum = cur_sum
                        best_rows = [list(r) for r in cur_rows]
                        best_diffs = [d[:] for d in cur_diffs]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    cur_diffs[i] = old_diffs
                    row_masks[i] = old_mask

            T = max(min_T, T * cooling)

        return best_scope, best_rows

    # --- Final polish: try to reduce scope by 1 repeatedly ---
    def final_polish(rows, deadline):
        cur = [list(r) for r in rows]
        n = len(cur)
        cur_masks = [row_mask(r) for r in cur]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_scope = scope_of(cur)
            for i in range(n):
                if time.time() >= deadline:
                    break
                cur_scope_i = cur[i][-1]
                if cur_scope_i <= 1:
                    continue
                used_wo = 0
                for ri in range(n):
                    if ri != i:
                        used_wo |= cur_masks[ri]
                # Try to find row with max < cur_scope_i
                res = find_row_dfs(used_wo, cur_scope_i, cur_scope_i - 1, 80000, min(deadline, time.time() + 1.0))
                if res is not None:
                    new_row, new_mask = res
                    if new_row[-1] < cur[i][-1]:
                        cur[i] = new_row
                        cur_masks[i] = new_mask
                        improved = True
                        break  # Restart loop with new scope
        return cur

    def run(seed, dl):
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        for _ in range(n_rows):
            if time.time() > dl:
                break
            res = find_row_dfs(used_mask, 200, 200, 100000, dl)
            if res is None:
                row = greedy_row(used_mask)
                used_mask |= row_mask(row)
            else:
                row, rm = res
                used_mask |= rm
            rows.append(row)
        while len(rows) < n_rows:
            row = greedy_row(used_mask)
            used_mask |= row_mask(row)
            rows.append(row)

        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)

        # SA refinement
        if time.time() < dl - 5:
            sa_scope, sa_rows = sa_optimize(rows, 5000, rng, dl - 2)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]

        # Final polish
        if time.time() < dl - 1:
            polished = final_polish(best_rows, dl - 1)
            if scope_of(polished) < best_scope:
                best_scope = scope_of(polished)
                best_rows = [r[:] for r in polished]

        return best_scope, best_rows

    try:
        overall = None
        seed = 0
        first = True
        while time.time() < DEADLINE - 5:
            if first:
                dl = min(DEADLINE - 3, time.time() + 40)
                first = False
            else:
                dl = min(DEADLINE - 3, time.time() + 12)
            sc, rows = run(seed, dl)
            if overall is None or sc < overall[0]:
                overall = (sc, rows)
            seed += 1
        if overall is not None:
            return overall[1]
    except Exception:
        pass

    # Fallback greedy
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used_mask)
        used_mask |= row_mask(row)
        rows.append(row)
    return rows
