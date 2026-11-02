"""Difference triangle set solver: Backtracking CSP row-finder + ILS."""
import time
import random
import sys


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Backtracking row finder with forward checking ----------
    def backtracking_row(used_other, cap, randomize=False, seed=None, max_nodes=30000):
        sys.setrecursionlimit(10000)
        node_count = [0]

        def dfs(pos, current_row, current_diffs):
            if node_count[0] > max_nodes:
                return None
            node_count[0] += 1

            if pos > k:
                return list(current_row)

            lo = current_row[-1] + 1
            hi = cap
            if lo > hi:
                return None

            candidates = []
            for x in range(lo, hi + 1):
                valid = True
                for j in range(pos):
                    d = x - current_row[j]
                    if d in used_other or d in current_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(x)

            if not candidates:
                return None

            if randomize:
                rng = random.Random(seed) if seed is not None else random
                rng.shuffle(candidates)

            for x in candidates:
                new_row = current_row + [x]
                new_diffs = set(current_diffs)
                for j in range(pos):
                    new_diffs.add(x - current_row[j])

                if pos < k:
                    fc_ok = False
                    for x_prime in range(x + 1, min(x + 40, cap + 1)):
                        fc_valid = True
                        for j in range(pos + 1):
                            d = x_prime - current_row[j]
                            if d in used_other or d in new_diffs:
                                fc_valid = False
                                break
                        if fc_valid:
                            fc_ok = True
                            break
                    if not fc_ok:
                        continue

                result = dfs(pos + 1, new_row, new_diffs)
                if result is not None:
                    return result

            return None

        return dfs(1, [0], set())

    # ---------- Greedy row ----------
    def greedy_row(used_set, jitter=0.0, cap=10**6):
        row, x = [0], 0
        row_diffs = set()
        while len(row) < k + 1:
            x += 1
            if x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set) and not (new & row_diffs):
                if jitter > 0 and random.random() < jitter and x < 400:
                    continue
                row.append(x)
                row_diffs |= new
        return row

    def used_set_except(rows, idx):
        used = set()
        for j in range(n_rows):
            if j != idx:
                used |= row_diffs_set(rows[j])
        return used

    def greedy_full(jitter=0.0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, jitter=jitter)
            if row is None:
                return None
            rows.append(row)
            used |= row_diffs_set(row)
        return rows

    def find_row_with_scope(used_other, target_scope, randomize=False):
        if randomize:
            seed = random.randint(0, 10**9)
        else:
            seed = None
        return backtracking_row(used_other, target_scope, randomize=randomize, seed=seed, max_nodes=25000)

    def try_reduce_row(rows, row_idx, used_other, max_attempts=20):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            target = best_max - 1
            if target < k:
                break
            randomize = attempt > 3
            row = find_row_with_scope(used_other, target, randomize=randomize)
            if row is not None:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def local_polish(rows, max_passes=8):
        rows = [list(r) for r in rows]
        improved = True
        passes = 0
        while improved and time_left() > 8 and passes < max_passes:
            improved = False
            passes += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 8:
                    break
                used_other = used_set_except(rows, i)
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=15)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    def element_level_polish(rows, max_time=12):
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for idx in order:
                if time.time() - start > max_time - 2:
                    break
                row = rows[idx]
                row_diffs = row_diffs_set(row)
                used_other = used_set_except(rows, idx)
                for pos in range(len(row) - 1, 0, -1):
                    orig = row[pos]
                    for delta in (1, 2, 3, 4, 5):
                        new_val = orig - delta
                        if new_val <= row[pos - 1]:
                            break
                        new_diffs = set()
                        ok = True
                        for i in range(pos):
                            d = new_val - row[i]
                            if d in used_other or d in row_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                        if not ok:
                            continue
                        if len(new_diffs) != len(new_diffs):
                            continue
                        row[pos] = new_val
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(orig - row[i])
                        row_diffs = row_diffs - old_diffs | new_diffs
                        improved = True
                        break
        return rows

    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = used_set_except(rows, idx)
            current_max = max(rows[idx])
            target = current_max + random.randint(0, 15)
            done = False
            for attempt in range(12):
                row = find_row_with_scope(used_other, target, randomize=True)
                if row is not None:
                    rows[idx] = row
                    done = True
                    break
            if not done:
                row = greedy_row(used_other, jitter=0.3, cap=current_max + 40)
                if row is not None:
                    rows[idx] = row
                else:
                    rows[idx] = list(rows[idx])
        return rows

    # ---------- Main search ----------
    best_rows = greedy_full(jitter=0.0)
    best_scope = scope(best_rows)

    if time_left() > 30:
        for r in range(10):
            if time_left() < 30:
                break
            cand = greedy_full(jitter=random.uniform(0.05, 0.35))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        if time_left() > 10:
            elem_polished = element_level_polish(best_rows, max_time=10)
            if scope(elem_polished) < best_scope:
                best_rows = elem_polished
                best_scope = scope(best_rows)

    while time_left() > 35:
        polished = local_polish(best_rows, max_passes=5)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        if time_left() > 25:
            elem_polished = element_level_polish(best_rows, max_time=6)
            if scope(elem_polished) < best_scope:
                best_rows = elem_polished
                best_scope = scope(best_rows)
        if time_left() < 30:
            break
        kicked = kick(best_rows)
        polished_kicked = local_polish(kicked, max_passes=5)
        if is_valid(polished_kicked):
            s = scope(polished_kicked)
            if s < best_scope:
                best_rows = polished_kicked
                best_scope = s
                if time_left() > 20:
                    elem_kicked = element_level_polish(polished_kicked, max_time=5)
                    if scope(elem_kicked) < best_scope:
                        best_rows = elem_kicked
                        best_scope = scope(best_rows)

    if time_left() > 20:
        for _ in range(3):
            if time_left() < 15:
                break
            elem_final = element_level_polish(best_rows, max_time=4)
            if scope(elem_final) < best_scope:
                best_rows = elem_final
                best_scope = scope(best_rows)
        if time_left() > 15:
            polished_final = local_polish(best_rows, max_passes=10)
            if scope(polished_final) < best_scope:
                best_rows = polished_final
                best_scope = scope(polished_final)
        if time_left() > 12:
            elem_last = element_level_polish(best_rows, max_time=3)
            if scope(elem_last) < best_scope:
                best_rows = elem_last
                best_scope = scope(best_rows)

    if not is_valid(best_rows):
        best_rows = greedy_full(jitter=0.0)

    return best_rows
