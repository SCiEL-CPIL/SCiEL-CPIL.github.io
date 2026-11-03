import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 143.0
    row_len = k + 1

    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def greedy_row(used_mask, max_scope=None, rng=None):
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            if rng is not None and rng.random() < 0.05:
                x += rng.randint(1, 2)
            if max_scope is not None and x > max_scope:
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
        for attempt in range(attempts):
            row = [0]
            x = 0
            local_mask = 0
            for j in range(1, row_len):
                found = False
                start = x + 1
                if attempt > 0 and rng.random() < 0.3:
                    start = x + rng.randint(1, 3)
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

        T = 5.0
        cooling = 0.9997
        min_T = 0.05

        for it in range(iters):
            if time.time() > deadline:
                break

            r = rng.random()

            if r < 0.35:
                # Row rebuild
                i = rng.randint(0, n - 1)
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                # Use a slightly larger scope to allow escape, but prefer compression
                rebuild_scope = cur_scope + 2
                res = row_rebuild_greedy(used_without, rebuild_scope, rng, attempts=5)
                if res is None:
                    continue
                new_row, new_mask = res

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

            elif r < 0.60:
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
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                # Try to find a tighter row via DFS with a small budget
                res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, 40000, min(deadline, time.time() + 1.5))
                if res is not None:
                    new_row, new_mask = res
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

            elif r < 0.80:
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
                # Swap two rows' contents (if they don't conflict with themselves)
                if n >= 2:
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 1)
                    if i == j:
                        continue
                    old_i = list(cur_rows[i])
                    old_j = list(cur_rows[j])
                    mask_i = row_masks[i]
                    mask_j = row_masks[j]
                    
                    # Check if swapping is valid: row i's diffs must not conflict with all rows except i,
                    # and row j's diffs must not conflict with all rows except j.
                    # Since we are swapping, the set of diffs used by the other rows remains the same.
                    # So we just need to check:
                    # 1. mask_j & (used_without_i) == 0  (where used_without_i is mask of all rows except i)
                    # 2. mask_i & (used_without_j) == 0
                    # But used_without_i = total_mask ^ mask_i
                    # used_without_j = total_mask ^ mask_j
                    # total_mask = OR of all row masks.
                    
                    # Actually, simpler:
                    # The global set of used diffs is the union of all row masks.
                    # When we swap, the global set of used diffs does NOT change.
                    # So we just need to ensure that the new assignment is valid.
                    # Since the set of diffs is the same, and each row's diffs are disjoint from the others 
                    # in the current state, swapping two rows preserves disjointness.
                    # Wait, no. If row i has diffs A and row j has diffs B, and A and B are disjoint from 
                    # all other rows' diffs, then swapping them keeps A in row j and B in row i.
                    # The condition is that A must be disjoint from all rows except i, and B from all except j.
                    # After swap, A is in row j, so A must be disjoint from all rows except j.
                    # Since A was disjoint from all rows except i, and i != j, A is disjoint from j's neighbors.
                    # But is A disjoint from j? No, A and B were disjoint, so A is disjoint from B.
                    # So the swap is always valid if the current state is valid.
                    # However, the cost might change if the scope changes? No, scope is max of all rows.
                    # So swapping doesn't change scope or conflicts. It's a neutral move.
                    # Let's skip this as it doesn't help.
                    pass

            T = max(min_T, T * cooling)

        return best_scope, best_rows

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
                if row is None:
                    row = greedy_row(used_mask, max_scope=500)
                    if row is None:
                        continue
                used_mask |= row_mask(row)
            else:
                row, rm = res
                used_mask |= rm
            rows.append(row)
        while len(rows) < n_rows:
            row = greedy_row(used_mask, max_scope=500)
            if row is None:
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
            else:
                used_mask |= row_mask(row)
            rows.append(row)

        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)

        # SA refinement
        if time.time() < dl - 5:
            sa_scope, sa_rows = sa_optimize(rows, k, 8000, rng, dl - 2)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]

        # Final polish: try to compress scope further via targeted DFS
        if time.time() < dl - 1:
            cur = [list(r) for r in best_rows]
            n = len(cur)
            cur_masks = [row_mask(r) for r in cur]
            improved = True
            while improved and time.time() < dl - 1:
                improved = False
                for i in range(n):
                    if time.time() >= dl - 1:
                        break
                    cur_scope_i = cur[i][-1]
                    # Try to find a row with smaller max
                    used_wo = 0
                    for ri in range(n):
                        if ri != i:
                            used_wo |= cur_masks[ri]
                    res = find_row_dfs(used_wo, cur_scope_i, cur_scope_i - 1, 80000, min(dl - 1, time.time() + 1.5))
                    if res is not None:
                        new_row, new_mask = res
                        if new_row[-1] < cur[i][-1]:
                            cur[i] = new_row
                            cur_masks[i] = new_mask
                            improved = True
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
                dl = min(DEADLINE - 3, time.time() + 50)
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

    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used_mask, max_scope=500)
        if row is None:
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
        else:
            used_mask |= row_mask(row)
        rows.append(row)
    return rows
