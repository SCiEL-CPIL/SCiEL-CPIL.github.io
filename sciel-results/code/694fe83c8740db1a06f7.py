import time
import random

def construct(n_rows, k):
    # n_rows=4, k=7 (8 numbers per row)
    # Minimize M (largest entry) subject to all positive differences being distinct.

    start_time = time.time()
    deadline = start_time + 135.0  # safety margin under the 150s kill

    def time_left():
        return deadline - time.time()

    # ---------- Difference helpers ----------
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

    # ---------- Greedy construction (deterministic / jittered, with cap) ----------
    def greedy(jitter=0.0, cap=None):
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                # window grows if we are tight on cap
                window = 120
                for trial_x in range(x + 1, x + window + 1):
                    if cap is not None and trial_x > cap:
                        break
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 400:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    # expand window
                    for trial_x in range(x + 1, x + 600 + 1):
                        if cap is not None and trial_x > cap:
                            break
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
                return None
        return rows

    # ---------- Backtracking row builder (MRV + forward checking) ----------
    # Builds a single row of size k+1 starting at 0, given the diffs already used
    # by other rows (used_other) and an optional cap. Returns a row or None.
    def backtrack_row(used_other, cap=None, max_nodes=40000):
        # We choose the next element greedily with backtracking.
        # State: current row, current row_diffs, next candidate start.
        best = [None]
        nodes = [0]

        def rec(row, row_diffs, x_start):
            if nodes[0] > max_nodes:
                return
            nodes[0] += 1
            if len(row) == k + 1:
                if best[0] is None or max(row) < max(best[0]):
                    best[0] = list(row)
                return
            # Try candidates from x_start upward
            lo = x_start
            hi = (cap if cap is not None else 2000)
            for trial_x in range(lo, hi + 1):
                if nodes[0] > max_nodes:
                    return
                new = {trial_x - a for a in row}
                if (new & used_other) or (new & row_diffs):
                    continue
                row.append(trial_x)
                row_diffs |= new
                # forward check: is there any valid next value? (light lookahead)
                rec(row, row_diffs, trial_x + 1)
                row.pop()
                row_diffs -= new
                if best[0] is not None and len(best[0]) == k + 1:
                    # we found a solution; keep searching a bit for a smaller max
                    # but stop if we already hit the theoretical minimum for this row
                    if max(best[0]) == len(row) + 1:
                        return

        rec([0], set(), 1)
        return best[0]

    # ---------- Per-row local optimization ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=120):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            # Alternate between jittered greedy and capped backtracking
            if attempt % 3 == 2:
                # capped backtracking: try to beat current_max
                cand = backtrack_row(used_other, cap=current_max - 1, max_nodes=20000)
                if cand is not None and max(cand) < best_max:
                    test_rows = [list(r) for r in rows]
                    test_rows[row_idx] = cand
                    if is_valid(test_rows):
                        best_row = list(cand)
                        best_max = max(cand)
                continue

            row, x = [0], 0
            current_row_diffs = set()
            success = True
            jitter = 0.08 + 0.06 * min(attempt / max_attempts, 1.0)
            window = 80 + 25 * min(attempt / max_attempts, 1.0)
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, int(x + window) + 1):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 300:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(x + 1, x + 250 + 1):
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

    def local_polish(rows, max_passes=6, attempts=80):
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
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=attempts)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    # ---------- Kick (conflict-aware rebuild) ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)

        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])

            success = False
            # First try a capped backtracking rebuild (tight)
            cand = backtrack_row(used_other, cap=max(rows[idx]) - 1, max_nodes=30000)
            if cand is not None:
                rows[idx] = cand
                success = True
            if not success:
                # Conflict-aware randomized rebuild
                for attempt in range(30):
                    row, x = [0], 0
                    current_row_diffs = set()
                    valid_row = True
                    jitter_prob = max(0.15, 0.5 - attempt * 0.012)
                    while len(row) < k + 1:
                        found = False
                        best_conflicts = float('inf')
                        best_conflict_x = None
                        best_conflict_new = None
                        window = 120
                        for trial_x in range(x + 1, x + window + 1):
                            new = {trial_x - a for a in row}
                            conflicts = len(new & used_other) + len(new & current_row_diffs)
                            if conflicts == 0:
                                if random.random() < jitter_prob and trial_x < 400:
                                    continue
                                row.append(trial_x)
                                current_row_diffs |= new
                                found = True
                                break
                            else:
                                if conflicts < best_conflicts:
                                    best_conflicts = conflicts
                                    best_conflict_x = trial_x
                                    best_conflict_new = new
                        if not found:
                            if best_conflict_x is not None and best_conflicts < 8:
                                row.append(best_conflict_x)
                                current_row_diffs |= best_conflict_new
                            else:
                                valid_row = False
                                break
                        x = row[-1]
                    if valid_row:
                        rows[idx] = row
                        success = True
                        break
            if not success:
                # Fallback greedy
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

    # ---------- Cap search: is there a solution with max entry <= M? ----------
    # Build rows under a hard cap M using greedy + backtracking.
    def build_under_cap(M, use_bt=True):
        rows = greedy(jitter=0.0, cap=M)
        if rows is not None:
            return rows
        # Try backtracking row by row
        used = set()
        rows = []
        for r_idx in range(n_rows):
            cand = backtrack_row(used, cap=M, max_nodes=60000)
            if cand is None:
                return None
            rows.append(cand)
            used |= row_diffs_set(cand)
        return rows

    # ---------- Main ----------
    # 1. Deterministic greedy baseline
    best_rows = greedy(jitter=0.0)
    best_scope = scope(best_rows)

    # 2. Cap search: probe a range of caps around the greedy scope to find a tighter feasible M
    if time_left() > 15:
        # Try to lower the cap a few units below the greedy scope
        for delta in range(0, 40):
            M = best_scope - delta
            if M < k:  # trivial lower bound: need at least k+1 distinct increasing ints
                break
            if time_left() < 5:
                break
            cand = build_under_cap(M)
            if cand is not None and scope(cand) <= M:
                best_rows = cand
                best_scope = scope(cand)
            else:
                # if we can't build under M, stop (monotonic-ish heuristic)
                # but allow a couple more probes to avoid a single unlucky failure
                if delta > 0 and delta % 5 == 0:
                    continue
                break

    # 3. Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # 4. Iterated Local Search
    consecutive_no_improve = 0
    max_no_improve = 25
    while time_left() > 20 and consecutive_no_improve < max_no_improve:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
            consecutive_no_improve = 0

        if time_left() < 15:
            break

        kicked_rows = kick(best_rows)
        polished_kicked = local_polish(kicked_rows)

        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    # 5. Final high-precision polish: tighter cap-driven backtracking per row
    if time_left() > 8:
        def final_polish(rows):
            rows = [list(r) for r in rows]
            improved = True
            max_passes = 10
            pass_count = 0
            while improved and time_left() > 4 and pass_count < max_passes:
                improved = False
                pass_count += 1
                order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
                for i in order:
                    if time_left() < 4:
                        break
                    used_other = set()
                    for j in range(n_rows):
                        if j != i:
                            used_other |= row_diffs_set(rows[j])
                    new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=200)
                    if new_max < max(rows[i]):
                        old_scope = scope(rows)
                        rows[i] = new_row
                        new_scope = scope(rows)
                        if new_scope < old_scope:
                            improved = True
            return rows

        polished = final_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # 6. Last-resort cap probe just below the current best to squeeze the final unit
    if time_left() > 3:
        for delta in range(1, 12):
            M = best_scope - delta
            if M < k:
                break
            if time_left() < 2:
                break
            cand = build_under_cap(M)
            if cand is not None and scope(cand) <= M:
                best_rows = cand
                best_scope = scope(cand)

    return best_rows
