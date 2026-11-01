"""Difference triangle set (4,7) - Greedy + ILS with element-level descent and targeted backtracking."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 142  # 8s safety margin under 150s kill

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
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    def total_diffs(rows):
        """All differences across all rows as a set."""
        s = set()
        for row in rows:
            s |= row_diffs_set(row)
        return s

    # ---------- Greedy baseline ----------
    def greedy(seed=0):
        rng = random.Random(seed) if seed > 0 else random
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                window = 100 if (seed > 0 and r_idx > 0) else 50
                for trial_x in range(x + 1, x + window):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        if seed > 0 and rng.random() < 0.08 and trial_x < 200:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(x + 1, x + 500):
                        new = {trial_x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            used |= new
                            found = True
                            break
                if not found:
                    break
                x = row[-1]
            if len(row) == k + 1:
                rows.append(row)
            else:
                rows = []
                break

        if len(rows) != n_rows:
            # Fallback: pure sequential greedy
            used = set()
            rows = []
            for _ in range(n_rows):
                row, x = [0], 0
                row_diffs = set()
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        row.append(x)
                        row_diffs |= new
                        used |= new
                rows.append(row)
        return rows

    # ---------- Element-level descent: try to lower each element ----------
    def element_descent(rows, max_time=2.0):
        """Try to reduce each element of each row to a smaller valid value."""
        rows = [list(r) for r in rows]
        t0 = time.time()
        improved = True
        while improved and time.time() - t0 < max_time:
            improved = False
            for r_idx in range(n_rows):
                if time.time() - t0 > max_time:
                    break
                row = rows[r_idx]
                # Compute used diffs from other rows
                used_other = set()
                for j in range(n_rows):
                    if j != r_idx:
                        used_other |= row_diffs_set(rows[j])
                # Try to reduce each element (from last to first to keep order)
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() - t0 > max_time:
                        break
                    current_val = row[pos]
                    # Lower bound: must be > row[pos-1]
                    lower = row[pos - 1] + 1
                    if current_val == lower:
                        continue
                    # Try values from lower to current_val - 1
                    for trial in range(lower, current_val):
                        # Check if this value works at position pos
                        new_diffs = set()
                        valid = True
                        for i in range(pos):
                            d = trial - row[i]
                            if d in used_other or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            # Also check diffs with elements after pos
                            for j in range(pos + 1, len(row)):
                                d = row[j] - trial
                                if d in used_other or d in new_diffs:
                                    valid = False
                                    break
                                new_diffs.add(d)
                            if valid:
                                row[pos] = trial
                                improved = True
                                # Recompute used_other? No, this row's diffs changed.
                                # But we're modifying in place; need to be careful.
                                # Actually, since we're reducing, the diffs change.
                                # Let's just break and restart the row.
                                break
                # After modifying a row, update used_other for subsequent rows
                # Actually, we recompute used_other each row, so it's fine.
        return rows

    # ---------- Row reconstruction with jitter ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        rng = random

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            jitter = 0.08 + 0.07 * min(attempt / max_attempts, 1.0)
            while len(row) < k + 1:
                found = False
                window = 80 + 30 * min(attempt / max_attempts, 1.0)
                for trial_x in range(x + 1, int(x + window)):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and rng.random() < jitter and trial_x < 300:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(x + 1, x + 300):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    # ---------- Local polish: row reconstruction + element descent ----------
    def local_polish(rows, max_passes=5):
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=60)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
            # Element-level descent after row reconstruction
            rows = element_descent(rows, max_time=3.0)
            if scope(rows) < scope([list(r) for r in rows]):
                pass
        return rows

    # ---------- Kick: perturb a random element ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 1, 2])
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))

        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])

            # Perturb one random element of the row
            row = list(rows[idx])
            pos = random.randint(1, len(row) - 1)
            # Try to increase the element a bit, then repair
            delta = random.randint(1, 15)
            row[pos] += delta
            # Now repair: try to reduce elements from pos onward
            # First, check if row is still strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    row[i] = row[i-1] + 1
            # Now try to reduce each element from pos to end
            for p in range(pos, len(row)):
                lower = row[p-1] + 1 if p > 0 else 1
                for trial in range(lower, row[p]):
                    # Check validity at position p
                    new_diffs = set()
                    valid = True
                    for i in range(p):
                        d = trial - row[i]
                        if d in used_other or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        for j in range(p + 1, len(row)):
                            d = row[j] - trial
                            if d in used_other or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                    if valid:
                        row[p] = trial
                        break
            # Verify the row is valid with respect to used_other
            row_diffs = row_diffs_set(row)
            if not (row_diffs & used_other):
                rows[idx] = row
            else:
                # Fallback: reconstruct the row
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # ---------- Backtracking with most-constrained-first row ordering ----------
    def backtrack_search(target_scope, time_limit):
        start = time.time()

        def solve_bt():
            rows_local = []
            used_local = set()
            nodes_visited = 0
            max_nodes = 800000

            # Determine row order: try to place rows that are "hardest" first
            # Heuristic: rows with larger required scope should go first
            # For simplicity, use fixed order but with randomization
            row_order = list(range(n_rows))

            def rec(row_idx, curr_row, curr_diffs):
                nonlocal nodes_visited
                if time.time() - start > time_limit or nodes_visited > max_nodes:
                    return False

                if len(curr_row) == k + 1:
                    rows_local.append(list(curr_row))
                    used_local.update(curr_diffs)

                    if row_idx == n_rows - 1:
                        return True

                    return rec(row_idx + 1, [0], set())

                last_val = curr_row[-1]
                candidates = list(range(last_val + 1, min(target_scope, last_val + 200) + 1))

                def conflict_score(val):
                    new_diffs = {val - a for a in curr_row}
                    conflicts = len(new_diffs & used_local) + len(new_diffs & curr_diffs)
                    return (conflicts, val)

                if random.random() < 0.25:
                    random.shuffle(candidates)
                else:
                    candidates.sort(key=conflict_score)

                max_candidates = min(len(candidates), 200)
                candidates = candidates[:max_candidates]

                for val in candidates:
                    if time.time() - start > time_limit or nodes_visited > max_nodes:
                        return False

                    nodes_visited += 1

                    new_diffs = set()
                    valid = True
                    for a in curr_row:
                        d = val - a
                        if d in used_local or d in curr_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                    if valid:
                        if rec(row_idx, curr_row + [val], curr_diffs | new_diffs):
                            return True

                return False

            if rec(0, [0], set()):
                # Reorder rows back to original order
                return rows_local
            return None

        return solve_bt()

    # ---------- Two-row swap: rebuild two rows simultaneously ----------
    def try_swap_rows(rows, i, j, max_attempts=40):
        """Try to find better versions of rows i and j simultaneously."""
        current_max = max(max(rows[i]), max(rows[j]))
        best_i = list(rows[i])
        best_j = list(rows[j])
        best_max = current_max

        used_other = set()
        for idx in range(n_rows):
            if idx != i and idx != j:
                used_other |= row_diffs_set(rows[idx])

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            # Try to build row i
            row_i, x = [0], 0
            diffs_i = set()
            success_i = True
            jitter = 0.10 + 0.05 * min(attempt / max_attempts, 1.0)
            while len(row_i) < k + 1:
                found = False
                window = 80 + 20 * min(attempt / max_attempts, 1.0)
                for trial_x in range(x + 1, int(x + window)):
                    new = {trial_x - a for a in row_i}
                    if not (new & used_other) and not (new & diffs_i):
                        if jitter > 0 and random.random() < jitter and trial_x < 300:
                            continue
                        row_i.append(trial_x)
                        diffs_i |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(x + 1, x + 250):
                        new = {trial_x - a for a in row_i}
                        if not (new & used_other) and not (new & diffs_i):
                            row_i.append(trial_x)
                            diffs_i |= new
                            found = True
                            break
                if not found:
                    success_i = False
                    break
                x = row_i[-1]

            if not success_i:
                continue

            # Try to build row j given row_i's diffs
            row_j, x = [0], 0
            diffs_j = set()
            success_j = True
            while len(row_j) < k + 1:
                found = False
                window = 80 + 20 * min(attempt / max_attempts, 1.0)
                for trial_x in range(x + 1, int(x + window)):
                    new = {trial_x - a for a in row_j}
                    if not (new & used_other) and not (new & diffs_i) and not (new & diffs_j):
                        if jitter > 0 and random.random() < jitter and trial_x < 300:
                            continue
                        row_j.append(trial_x)
                        diffs_j |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(x + 1, x + 250):
                        new = {trial_x - a for a in row_j}
                        if not (new & used_other) and not (new & diffs_i) and not (new & diffs_j):
                            row_j.append(trial_x)
                            diffs_j |= new
                            found = True
                            break
                if not found:
                    success_j = False
                    break
                x = row_j[-1]

            if success_j:
                new_max = max(max(row_i), max(row_j))
                if new_max < best_max:
                    test_rows = [list(r) for r in rows]
                    test_rows[i] = row_i
                    test_rows[j] = row_j
                    if is_valid(test_rows):
                        best_i = list(row_i)
                        best_j = list(row_j)
                        best_max = new_max

        return best_i, best_j, best_max

    # ---------- Main search ----------
    best_rows = greedy()
    best_scope = scope(best_rows)

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows, max_passes=4)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Phase 1: Randomized Greedy Restarts
    num_restarts = 12
    for i in range(num_restarts):
        if time_left() < 25:
            break

        new_rows = greedy(seed=i + 1)
        if new_rows:
            new_scope = scope(new_rows)
            if new_scope < best_scope:
                best_rows = new_rows
                best_scope = new_scope
                if time_left() > 15:
                    polished = local_polish(best_rows, max_passes=3)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)

    # Phase 2: ILS with element-level descent
    ils_count = 0
    while time_left() > 35 and ils_count < 50:
        ils_count += 1
        polished = local_polish(best_rows, max_passes=3)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        if time_left() < 30:
            break

        kicked_rows = kick(best_rows)
        polished_kicked = local_polish(kicked_rows, max_passes=3)

        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Phase 3: Two-row swaps
    if time_left() > 25:
        for _ in range(10):
            if time_left() < 20:
                break
            i, j = random.sample(range(n_rows), 2)
            new_i, new_j, new_max = try_swap_rows(best_rows, i, j, max_attempts=30)
            if new_max < max(max(best_rows[i]), max(best_rows[j])):
                test_rows = [list(r) for r in best_rows]
                test_rows[i] = new_i
                test_rows[j] = new_j
                if is_valid(test_rows):
                    best_rows = test_rows
                    best_scope = scope(best_rows)

    # Phase 4: Targeted Backtracking
    if time_left() > 18:
        target = best_scope - 1
        bt_rows = backtrack_search(target, time_limit=12)
        if bt_rows:
            best_rows = bt_rows
            best_scope = target
            if time_left() > 8:
                polished = local_polish(best_rows, max_passes=3)
                if scope(polished) < best_scope:
                    best_rows = polished
                    best_scope = scope(polished)

        if time_left() > 15:
            target = best_scope - 1
            bt_rows = backtrack_search(target, time_limit=10)
            if bt_rows:
                best_rows = bt_rows
                best_scope = target
                if time_left() > 6:
                    polished = local_polish(best_rows, max_passes=2)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)

        if time_left() > 12:
            target = best_scope - 1
            bt_rows = backtrack_search(target, time_limit=8)
            if bt_rows:
                best_rows = bt_rows
                best_scope = target
                if time_left() > 5:
                    polished = local_polish(best_rows, max_passes=2)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)

    # Final high-precision polish: exhaustive element-level sweep
    if time_left() > 8:
        best_rows = [list(r) for r in best_rows]
        for _ in range(3):
            if time_left() < 3:
                break
            improved = False
            order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
            for i in order:
                if time_left() < 2:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(best_rows[j])
                new_row, new_max = try_reduce_row(best_rows, i, used_other, max_attempts=150)
                if new_max < max(best_rows[i]):
                    best_rows[i] = new_row
                    improved = True
            # Element-level descent
            polished = element_descent(best_rows, max_time=4.0)
            if scope(polished) < scope(best_rows):
                best_rows = polished
                improved = True
            if not improved:
                break

        if scope(best_rows) < best_scope:
            best_scope = scope(best_rows)

    # Final validity guarantee.
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
