import time
import random
import math
import sys

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 138.0
    m = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def time_ok(margin=5.0):
        return time.time() < deadline - margin

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    # ---- Bitmask helpers ----
    def make_bm(diffs, max_d):
        bm = 0
        for d in diffs:
            if d <= max_d:
                bm |= (1 << d)
        return bm

    def bm_conflict(bm, diffs):
        for d in diffs:
            if bm & (1 << d):
                return True
        return False

    # ---- Fast row generation with bitmask ----
    def gen_row_fast(used_bm, max_val, k_val, rng, p_small=0.7, window=3):
        row = [0]
        cur_bm = 0
        for _ in range(k_val):
            best_cand = None
            count = 0
            x = row[-1] + 1
            while x <= max_val:
                new_diffs = [x - a for a in row]
                conflict = False
                for d in new_diffs:
                    if (used_bm | cur_bm) & (1 << d):
                        conflict = True
                        break
                if not conflict and len(set(new_diffs)) == len(new_diffs):
                    if best_cand is None:
                        best_cand = x
                    count += 1
                    if count >= window:
                        break
                x += 1
            if best_cand is None:
                return None
            if count > 1 and rng.random() >= p_small:
                # pick a random one from the window
                # re-scan to collect
                cands = []
                x = row[-1] + 1
                while x <= max_val and len(cands) < window:
                    new_diffs = [x - a for a in row]
                    conflict = False
                    for d in new_diffs:
                        if (used_bm | cur_bm) & (1 << d):
                            conflict = True
                            break
                    if not conflict and len(set(new_diffs)) == len(new_diffs):
                        cands.append(x)
                    x += 1
                nxt = rng.choice(cands)
            else:
                nxt = best_cand
            row.append(nxt)
            for a in row[:-1]:
                cur_bm |= (1 << (nxt - a))
        return row

    # ---- Tighten: pull elements down greedily ----
    def tighten(rows, max_iter=500):
        rows = [list(r) for r in rows]
        for _ in range(max_iter):
            if not time_ok(15.0):
                break
            improved = False
            for r in range(n_rows):
                row = rows[r]
                for pos in range(len(row) - 1, 0, -1):
                    target = row[pos - 1] + 1
                    if target >= row[pos]:
                        continue
                    cand = row[:pos] + [target] + row[pos + 1:]
                    ok = True
                    for i in range(1, len(cand)):
                        if cand[i] <= cand[i - 1]:
                            ok = False
                            break
                    if not ok:
                        continue
                    idiffs = set()
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            d = cand[j] - cand[i]
                            if d in idiffs:
                                ok = False
                                break
                            idiffs.add(d)
                        if not ok:
                            break
                    if not ok:
                        continue
                    for idx in range(n_rows):
                        if idx == r:
                            continue
                        od = row_diffs_set(rows[idx])
                        if idiffs & od:
                            ok = False
                            break
                    if ok:
                        row[pos] = target
                        improved = True
            if not improved:
                break
        return rows

    # ---- Greedy initial construction ----
    def greedy_initial(rng, max_val=400):
        used_bm = 0
        rows = []
        for r in range(n_rows):
            row = None
            for attempt in range(150):
                row = gen_row_fast(used_bm, max_val, k, rng, p_small=0.85, window=3)
                if row is not None:
                    break
            if row is None:
                row = gen_row_fast(used_bm, max_val + 250, k, rng, p_small=0.92, window=2)
            if row is None:
                return None
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_bm |= (1 << (row[j] - row[i]))
        return rows

    # ---- Backtracking with constraint propagation and MRV ----
    def backtrack_scope(target_scope, rng, max_nodes=100000):
        used_bm = 0
        rows = []
        nodes = [0]
        max_d = target_scope

        def build_row(row_idx, pos, cur_row, cur_bm_row):
            if not time_ok(10.0):
                return False
            nodes[0] += 1
            if nodes[0] > max_nodes:
                return False

            if pos == m:
                rows.append(cur_row)
                return True

            min_x = cur_row[-1] + 1 if pos > 0 else 1
            max_x = target_scope
            # Remaining positions: m - pos - 1, each needs at least +1
            remaining = m - pos - 1
            if max_x < min_x + remaining:
                return False

            valid = []
            for x in range(min_x, max_x + 1):
                new_diffs = [x - a for a in cur_row]
                conflict = False
                for d in new_diffs:
                    if d > max_d:
                        conflict = True
                        break
                    if (used_bm | cur_bm_row) & (1 << d):
                        conflict = True
                        break
                if conflict:
                    continue
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                valid.append(x)
                if len(valid) > 50:
                    break

            if not valid:
                return False

            # Prefer smaller values with some randomness
            if rng.random() < 0.75:
                pass  # already in order
            else:
                rng.shuffle(valid)

            for x in valid:
                new_row = cur_row + [x]
                new_bm = cur_bm_row
                for a in cur_row:
                    new_bm |= (1 << (x - a))
                if build_row(row_idx, pos + 1, new_row, new_bm):
                    return True
            return False

        for r in range(n_rows):
            if not build_row(r, 0, [0], 0):
                return None
        return rows

    # ---- Local refinement: move individual elements ----
    def local_refine(rows, rng, max_iter=10000):
        rows = [list(r) for r in rows]
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        for _ in range(max_iter):
            if not time_ok(10.0):
                break
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            for delta in [-1, 1, -2, 2, -3, 3]:
                new_val = row[pos] + delta
                if new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                test_row = row[:]
                test_row[pos] = new_val
                idiffs = set()
                ok = True
                for i in range(len(test_row)):
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        if d in idiffs:
                            ok = False
                            break
                        idiffs.add(d)
                    if not ok:
                        break
                if not ok:
                    continue
                conflict = False
                for idx in range(n_rows):
                    if idx == r_idx:
                        continue
                    if idiffs & row_diffs_set(rows[idx]):
                        conflict = True
                        break
                if conflict:
                    continue
                other_max = max(rows[i][-1] for i in range(n_rows) if i != r_idx)
                new_scope = max(other_max, test_row[-1])
                if new_scope <= current_max:
                    rows[r_idx] = test_row
                    current_max = new_scope
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows]
                    break
        return best_rows

    # ---- Simulated annealing with row regeneration ----
    def anneal(rows, rng, max_iter=15000):
        rows = [list(r) for r in rows]
        used_bm = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used_bm |= (1 << (r[j] - r[i]))
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        T = 8.0
        cooling = 0.999
        min_T = 0.1

        for _ in range(max_iter):
            if not time_ok(12.0):
                break
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    used_bm &= ~(1 << (old_row[j] - old_row[i]))

            p_sm = 0.55 + 0.4 * rng.random()
            win = 2 + rng.randint(0, 4)
            new_row = gen_row_fast(used_bm, current_max + 25, k, rng, p_small=p_sm, window=win)

            if new_row is None:
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        used_bm |= (1 << (old_row[j] - old_row[i]))
                continue

            new_diffs_set = row_diffs_set(new_row)
            other_max = max(rows[i][-1] for i in range(n_rows) if i != r_idx)
            new_global = max(other_max, new_row[-1])
            delta = new_global - current_max

            accept = False
            if delta <= 0:
                accept = True
            else:
                prob = math.exp(-delta / max(T, 0.01))
                accept = rng.random() < prob

            if accept:
                rows[r_idx] = new_row
                for d in new_diffs_set:
                    used_bm |= (1 << d)
                current_max = new_global
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            else:
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        used_bm |= (1 << (old_row[j] - old_row[i]))

            T = max(min_T, T * cooling)

        return best_rows

    # ---- Main search ----
    rng = random.Random(42)

    # 1. Greedy initial
    best_rows = greedy_initial(rng, max_val=350)
    if best_rows is None or not is_valid(best_rows):
        best_rows = greedy_initial(rng, max_val=500)
    if best_rows is None or not is_valid(best_rows):
        used_bm = 0
        best_rows = []
        for _ in range(n_rows):
            row = gen_row_fast(used_bm, 800, k, rng, p_small=0.95, window=2)
            if row is None:
                row = [0] + list(range(1, k + 1)) * 10
            best_rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_bm |= (1 << (row[j] - row[i]))

    # 2. Tighten
    best_rows = tighten(best_rows)
    if not is_valid(best_rows):
        best_rows = greedy_initial(rng, max_val=400)
        best_rows = tighten(best_rows)
    best_scope = scope_of(best_rows)

    # 3. Anneal
    rows_a = anneal(best_rows, rng, max_iter=12000)
    if is_valid(rows_a):
        s = scope_of(rows_a)
        if s < best_scope:
            best_rows = rows_a
            best_scope = s

    # 4. Local refine
    rows_l = local_refine(best_rows, rng, max_iter=6000)
    if is_valid(rows_l):
        s = scope_of(rows_l)
        if s < best_scope:
            best_rows = rows_l
            best_scope = s

    # 5. Tighten again
    best_rows = tighten(best_rows)
    if is_valid(best_rows):
        best_scope = scope_of(best_rows)

    # 6. Backtracking for lower scope
    target = best_scope - 1
    max_bt = 30
    bt_count = 0
    while target > 100 and bt_count < max_bt and time_ok(35.0):
        bt_count += 1
        rng_bt = random.Random(5000 + bt_count * 31337)
        time_left = deadline - 35 - time.time()
        est_nodes = max(8000, min(80000, int(time_left * 500)))
        sol = backtrack_scope(target, rng_bt, max_nodes=est_nodes)
        if sol and is_valid(sol):
            new_scope = scope_of(sol)
            if new_scope < best_scope:
                best_rows = sol
                best_scope = new_scope
                best_rows = tighten(best_rows)
                if is_valid(best_rows):
                    best_scope = scope_of(best_rows)
                rows_ref = local_refine(best_rows, rng_bt, max_iter=3000)
                if is_valid(rows_ref):
                    s_ref = scope_of(rows_ref)
                    if s_ref < best_scope:
                        best_rows = rows_ref
                        best_scope = s_ref
                target = best_scope - 1
            else:
                target -= 1
        else:
            target -= 1

    # 7. Diversified restarts
    for trial in range(8):
        if not time_ok(25.0):
            break
        rng2 = random.Random(2000 + trial * 7919)
        rows = greedy_initial(rng2, max_val=280 + trial * 25)
        if rows is None or not is_valid(rows):
            continue
        rows = tighten(rows)
        if not is_valid(rows):
            continue
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
        rows2 = anneal(rows, rng2, max_iter=8000)
        if is_valid(rows2):
            s2 = scope_of(rows2)
            if s2 < best_scope:
                best_rows = rows2
                best_scope = s2
        rows3 = local_refine(best_rows, rng2, max_iter=3000)
        if is_valid(rows3):
            s3 = scope_of(rows3)
            if s3 < best_scope:
                best_rows = rows3
                best_scope = s3

    # Final tighten
    best_rows = tighten(best_rows, max_iter=300)
    if not is_valid(best_rows):
        best_rows = greedy_initial(random.Random(7), max_val=500)
        best_rows = tighten(best_rows)

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
