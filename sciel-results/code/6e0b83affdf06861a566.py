import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 112.0
    row_len = k + 1

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
        counts = Counter()
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

    def row_rebuild_greedy(used_mask, max_scope, rng, attempts=3):
        best_row = None
        best_max = max_scope + 1
        
        for attempt in range(attempts):
            row = [0]
            x = 0
            local_mask = 0
            success = True
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
                    success = False
                    break
            if success:
                row_max = row[-1]
                if row_max < best_max:
                    best_max = row_max
                    best_row = row[:]
        
        if best_row is not None:
            return best_row, row_mask(best_row)
        return None

    def sa_optimize(rows, k, iters, rng, deadline):
        n = len(rows)
        cur_rows = [list(r) for r in rows]
        cur_scope = scope_of(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_scope = cur_scope

        diff_counts = build_diff_counts(cur_rows)
        row_masks = [row_mask(r) for r in cur_rows]

        def compute_used_without(i):
            u = 0
            for ri in range(n):
                if ri != i:
                    u |= row_masks[ri]
            return u

        def get_cost(rows, dc):
            scope = scope_of(rows)
            collisions = sum(max(0, c - 1) for c in dc.values())
            return scope + 1000 * collisions

        cur_cost = get_cost(cur_rows, diff_counts)
        best_cost = cur_cost

        T = 8.0
        cooling = 0.9995
        min_T = 0.03

        for it in range(iters):
            if time.time() > deadline:
                break

            r = rng.random()

            if r < 0.30:
                # Row rebuild with multiple attempts
                i = rng.randint(0, n - 1)
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                best_new_row = None
                best_new_mask = None
                best_new_max = old_row[-1]
                
                for attempt in range(3):
                    max_scope_try = cur_scope + (3 - attempt)
                    res = row_rebuild_greedy(used_without, max_scope_try, rng, attempts=2)
                    if res is not None:
                        new_row, new_mask = res
                        if new_row[-1] < best_new_max:
                            best_new_row = new_row
                            best_new_mask = new_mask
                            best_new_max = new_row[-1]
                
                if best_new_row is None:
                    continue

                new_row = best_new_row
                new_mask = best_new_mask

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                row_masks[i] = new_mask
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
                    row_masks[i] = old_mask

            elif r < 0.55:
                # Scope compression: target row with largest element
                # Sort rows by last element descending, pick from top half
                sorted_idx = sorted(range(n), key=lambda x: -cur_rows[x][-1])
                top_half = sorted_idx[:max(1, n // 2)]
                i = top_half[rng.randint(0, len(top_half) - 1)]
                
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                # Try to find a tighter row via DFS with adaptive budget
                time_budget = min(deadline - time.time(), 2.5)
                node_budget = min(80000, int(time_budget * 25000))
                res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, node_budget, min(deadline, time.time() + time_budget))
                if res is not None:
                    new_row, new_mask = res
                    if new_row[-1] < old_row[-1]:
                        remove_row_diffs(diff_counts, old_row)
                        add_row_diffs(diff_counts, new_row)
                        cur_rows[i] = new_row
                        row_masks[i] = new_mask
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
                            row_masks[i] = old_mask

            elif r < 0.75:
                # Single element move
                i = rng.randint(0, n - 1)
                j = rng.randint(1, row_len - 1)
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                lo = old_row[j - 1] + 1
                hi = old_row[j + 1] - 1 if j < row_len - 1 else cur_scope + 2
                if lo >= hi:
                    continue
                
                # Bias toward smaller values for compression
                bias_factor = rng.random()
                if bias_factor < 0.6:
                    span = hi - lo + 1
                    new_val = lo + int(span * rng.random() ** 1.5)
                else:
                    new_val = rng.randint(lo, hi)
                
                if new_val == old_row[j]:
                    continue

                new_row = list(old_row)
                new_row[j] = new_val
                if new_row[j] <= new_row[j-1] or (j < row_len-1 and new_row[j] >= new_row[j+1]):
                    continue

                new_mask = row_mask(new_row)
                if new_mask & used_without:
                    continue

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                row_masks[i] = new_mask
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
                    row_masks[i] = old_mask

            else:
                # Two-element coordinated move
                i = rng.randint(0, n - 1)
                j1 = rng.randint(1, row_len - 1)
                j2 = rng.randint(1, row_len - 1)
                if j1 == j2:
                    continue
                if j1 > j2:
                    j1, j2 = j2, j1
                    
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                lo1 = old_row[j1 - 1] + 1
                hi1 = old_row[j1 + 1] - 1 if j1 < row_len - 1 else cur_scope + 2
                lo2 = old_row[j2 - 1] + 1
                hi2 = old_row[j2 + 1] - 1 if j2 < row_len - 1 else cur_scope + 2
                
                if lo1 >= hi1 or lo2 >= hi2:
                    continue
                
                # Ensure ordering constraints
                if j2 == j1 + 1:
                    hi1 = min(hi1, old_row[j2] - 1)
                    lo2 = max(lo2, old_row[j1] + 1)
                elif j1 == j2 - 1:
                    hi2 = min(hi2, old_row[j1] - 1)
                    lo1 = max(lo1, old_row[j2] + 1)
                
                if lo1 >= hi1 or lo2 >= hi2:
                    continue
                
                new_val1 = rng.randint(lo1, hi1)
                new_val2 = rng.randint(lo2, hi2)
                
                if new_val1 == old_row[j1] and new_val2 == old_row[j2]:
                    continue

                new_row = list(old_row)
                new_row[j1] = new_val1
                new_row[j2] = new_val2
                
                # Verify strict increase
                valid = True
                for idx in range(1, row_len):
                    if new_row[idx] <= new_row[idx - 1]:
                        valid = False
                        break
                
                if not valid:
                    continue

                new_mask = row_mask(new_row)
                if new_mask & used_without:
                    continue

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                row_masks[i] = new_mask
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
                    row_masks[i] = old_mask

            T = max(min_T, T * cooling)

        return best_scope, best_rows

    def run(seed, dl):
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        for _ in range(n_rows):
            if time.time() > dl:
                break
            # Adaptive DFS budget based on remaining time
            remaining = dl - time.time()
            node_budget = min(100000, max(20000, int(remaining * 2000)))
            res = find_row_dfs(used_mask, 200, 200, node_budget, dl)
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

        # SA refinement with adaptive iteration count
        if time.time() < dl - 5:
            remaining_time = dl - 2 - time.time()
            # Scale iterations based on available time
            sa_iters = min(8000, max(2000, int(remaining_time * 100)))
            sa_scope, sa_rows = sa_optimize(rows, k, sa_iters, rng, dl - 2)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]

        # Final polish: iterative scope compression with adaptive budgets
        if time.time() < dl - 1:
            cur = [list(r) for r in best_rows]
            n = len(cur)
            cur_masks = [row_mask(r) for r in cur]
            
            # Sort indices by last element descending for priority
            for polish_round in range(3):
                if time.time() >= dl - 1:
                    break
                sorted_idx = sorted(range(n), key=lambda x: -cur[x][-1])
                improved = False
                for i in sorted_idx:
                    if time.time() >= dl - 1:
                        break
                    cur_scope_i = cur[i][-1]
                    # Try to find a row with smaller max
                    used_wo = 0
                    for ri in range(n):
                        if ri != i:
                            used_wo |= cur_masks[ri]
                    
                    # Adaptive time budget based on remaining time
                    remaining = dl - 1 - time.time()
                    time_budget = min(1.0, remaining * 0.3)
                    node_budget = min(60000, int(time_budget * 20000))
                    res = find_row_dfs(used_wo, cur_scope_i, cur_scope_i - 1, node_budget, min(dl - 1, time.time() + time_budget))
                    if res is not None:
                        new_row, new_mask = res
                        if new_row[-1] < cur[i][-1]:
                            cur[i] = new_row
                            cur_masks[i] = new_mask
                            improved = True
                
                if not improved:
                    break
            
            if scope_of(cur) < best_scope:
                best_scope = scope_of(cur)
                best_rows = [r[:] for r in cur]

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
                dl = min(DEADLINE - 3, time.time() + 8)
            sc, rows = run(seed, dl)
            if overall is None or sc < overall[0]:
                overall = (sc, rows)
            seed += 1
        if overall is not None:
            return overall[1]
    except Exception:
        pass

    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used_mask)
        used_mask |= row_mask(row)
        rows.append(row)
    return rows
