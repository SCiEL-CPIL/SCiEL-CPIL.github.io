"""Difference triangle set (4,7): bitmask-accelerated search with constraint propagation."""
import time
import random
import math
import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
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

    # ---------- Greedy baseline ----------
    def greedy(seed=0):
        if seed > 0:
            random.seed(seed)
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                window = 150 if (seed > 0 and r_idx > 0) else 80
                for trial_x in range(row[-1] + 1, row[-1] + window):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        if seed > 0 and random.random() < 0.08 and trial_x < 200:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(row[-1] + 1, row[-1] + 500):
                        new = {trial_x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            used |= new
                            found = True
                            break
                if not found:
                    break
            if len(row) == k + 1:
                rows.append(row)
        if len(rows) != n_rows:
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                row_diffs = set()
                while len(row) < k + 1:
                    row[-1] += 1
                    new = {row[-1] - a for a in row[:-1]}
                    if not (new & used) and not (new & row_diffs):
                        row.append(row[-1])
                        row_diffs |= new
                        used |= new
                rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Local optimization ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            row = [0]
            row_diffs = set()
            success = True
            jitter = 0.08 + 0.04 * (attempt / max_attempts)
            while len(row) < k + 1:
                found = False
                window = 60 + 30 * (attempt / max_attempts)
                for trial_x in range(row[-1] + 1, int(row[-1] + window)):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs):
                        if random.random() < jitter and trial_x < 250:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(row[-1] + 1, row[-1] + 200):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            found = True
                            break
                if not found:
                    success = False
                    break
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def local_polish(rows):
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        max_passes = 8
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
                        for d in row_diffs_list(rows[j]):
                            used_other.add(d)
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=120)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    for d in row_diffs_list(rows[j]):
                        used_other.add(d)
            row = [0]
            row_diffs = set()
            success = False
            for attempt in range(20):
                row = [0]
                row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(row[-1] + 1, row[-1] + 100):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & row_diffs):
                            if random.random() < 0.4 and trial_x < 300:
                                continue
                            row.append(trial_x)
                            row_diffs |= new
                            found = True
                            break
                    if not found:
                        valid_row = False
                        break
                if valid_row:
                    rows[idx] = row
                    success = True
                    break
            if not success:
                greedy_row = [0]
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    greedy_row[-1] += 1
                    new = {greedy_row[-1] - a for a in greedy_row[:-1]}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(greedy_row[-1])
                        row_diffs |= new
                rows[idx] = greedy_row
        return rows

    # ---------- Deterministic backtracking with gap variables and forward checking ----------
    def backtrack_search(target_scope, time_limit):
        start = time.time()
        rows_local = []
        used_local = set()
        nodes_visited = 0
        max_nodes = 500000

        # Precompute all possible differences up to target_scope
        all_diffs = set(range(1, target_scope + 1))

        def count_available_diffs(curr_diffs, row_so_far, target):
            """Count how many valid differences remain for filling the rest of the row."""
            remaining_positions = (k + 1) - len(row_so_far)
            # Each new element adds (len(row_so_far)) new differences
            # Total new differences needed = sum of 1, 2, ..., remaining_positions
            total_needed = sum(range(1, remaining_positions + 1))
            # Available diffs are those not in used_local and not in curr_diffs
            available = all_diffs - used_local - curr_diffs
            return len(available) >= total_needed

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

            # Forward checking: ensure enough differences remain
            if not count_available_diffs(curr_diffs, curr_row, target_scope):
                return False

            last_val = curr_row[-1]
            # Generate candidates: values that would keep the row within target_scope
            # and maintain increasing order
            min_next = last_val + 1
            max_next = target_scope
            candidates = list(range(min_next, max_next + 1))

            # Dynamic variable ordering: try values that leave the most flexibility
            # Score by: (1) number of conflicts with used_local and curr_diffs,
            #           (2) how many future differences become unavailable
            def flexibility_score(val):
                new_diffs = {val - a for a in curr_row}
                conflicts = len(new_diffs & used_local) + len(new_diffs & curr_diffs)
                if conflicts > 0:
                    return (conflicts, val)
                # Count how many of these new diffs are "large" (limit future options less)
                large_diffs = sum(1 for d in new_diffs if d > target_scope // 2)
                return (0, -large_diffs, val)

            # Sort by flexibility (fewer conflicts first, then more large diffs first)
            candidates.sort(key=flexibility_score)

            # Limit candidates to keep search tractable
            max_candidates = min(len(candidates), 80)
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
            return rows_local
        return None

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Multi-restart
    for i in range(12):
        if time_left() < 25:
            break
        new_rows = greedy(seed=i + 1)
        if new_rows:
            new_scope = scope(new_rows)
            if new_scope < best_scope:
                best_rows = new_rows
                best_scope = new_scope
                if time_left() > 15:
                    polished = local_polish(best_rows)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)

    # ILS
    while time_left() > 35:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        if time_left() < 28:
            break
        kicked_rows = kick(best_rows)
        polished_kicked = local_polish(kicked_rows)
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Targeted backtracking with increased time budget
    for improvement in range(1, 5):
        if time_left() < 12 - improvement * 2:
            break
        target = best_scope - improvement
        # Allocate more time for earlier improvements
        time_alloc = max(8, 18 - improvement * 3)
        bt_rows = backtrack_search(target, time_limit=time_alloc)
        if bt_rows:
            best_rows = bt_rows
            best_scope = target
            if time_left() > 5:
                polished = local_polish(best_rows)
                if scope(polished) < best_scope:
                    best_rows = polished
                    best_scope = scope(polished)

    # Final high-precision polish with tighter tolerance
    if time_left() > 8:
        best_rows = [list(r) for r in best_rows]
        # More iterations with tighter jitter
        for _ in range(6):
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
                        for d in row_diffs_list(best_rows[j]):
                            used_other.add(d)
                # More attempts with lower jitter for precision
                new_row, new_max = try_reduce_row(best_rows, i, used_other, max_attempts=400)
                if new_max < max(best_rows[i]):
                    best_rows[i] = new_row
                    improved = True
            if not improved:
                break
        if scope(best_rows) < best_scope:
            best_scope = scope(best_rows)

    # Ultra-final polish: try to shave off 1 more unit
    if time_left() > 5 and best_scope > 120:
        for i in range(n_rows):
            if time_left() < 2:
                break
            used_other = set()
            for j in range(n_rows):
                if j != i:
                    for d in row_diffs_list(best_rows[j]):
                        used_other.add(d)
            # Very aggressive reduction attempt
            new_row, new_max = try_reduce_row(best_rows, i, used_other, max_attempts=600)
            if new_max < max(best_rows[i]):
                best_rows[i] = new_row
        if scope(best_rows) < best_scope:
            best_scope = scope(best_rows)

    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
