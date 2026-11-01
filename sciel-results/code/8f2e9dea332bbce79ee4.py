"""Rewritten program for difference triangle set (4,7)."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_set(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline ----------
    def greedy():
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

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Backtracking row construction with pruning ----------
    def build_row_backtrack(used_other, max_val, depth=0, row=None, row_diffs=None):
        """
        Build a row of k+1 integers starting at 0, strictly increasing,
        with all differences distinct from used_other and within the row.
        Returns the row if found, else None.
        """
        if row is None:
            row = [0]
            row_diffs = set()
        if len(row) == k + 1:
            return row
        last = row[-1]
        # Try candidates from last+1 up to max_val
        for x in range(last + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used_other or new_diffs & row_diffs:
                continue
            # Pruning: check if remaining positions can fit
            remaining = k + 1 - len(row)
            # Minimum possible next values: x+1, x+2, ..., x+remaining
            # Need to ensure enough room
            if x + remaining > max_val:
                continue
            # Recursive call
            result = build_row_backtrack(used_other, max_val, depth + 1, row + [x], row_diffs | new_diffs)
            if result is not None:
                return result
        return None

    # ---------- Local optimization: try to reduce a specific row ----------
    def try_reduce_row(rows, row_idx, max_attempts=50):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            # Compute used differences from other rows
            used_other = set()
            for j in range(n_rows):
                if j != row_idx:
                    used_other |= row_diffs_set(rows[j])

            # Try backtracking with a slightly reduced max
            target_max = best_max - 1
            if target_max < k:
                break

            # Randomize the search order slightly
            row = build_row_backtrack(used_other, target_max)
            if row is not None:
                best_row = row
                best_max = max(row)
                if best_max == k:  # Can't do better than k
                    break
            else:
                # Try a randomized greedy approach
                row, x = [0], 0
                current_row_diffs = set()
                success = True
                jitter = 0.1 if attempt > 10 else 0.0
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, min(x + 60, target_max + 1)):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            if jitter > 0 and random.random() < jitter and trial_x < 200:
                                continue
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                    if not found:
                        success = False
                        break
                    x = row[-1]
                if success and max(row) < best_max:
                    best_row = row
                    best_max = max(row)
                    if best_max == k:
                        break

        return best_row, best_max

    # ---------- Local polish: reduce scope by re-optimizing rows ----------
    def local_polish(rows, max_passes=4):
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Order by current max, largest first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                new_row, new_max = try_reduce_row(rows, i, max_attempts=40)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    # ---------- Kick: perturb solution ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)

        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])

            # Rebuild row with high randomness
            success = False
            for attempt in range(20):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, x + 100):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            if random.random() < 0.4 and trial_x < 300:
                                continue
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                    if not found:
                        valid_row = False
                        break
                    x = row[-1]
                if valid_row:
                    rows[idx] = row
                    success = True
                    break

            if not success:
                # Fallback to greedy row
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

    # ---------- Main search loop ----------
    # Initial polish
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    while time_left() > 25:
        # Local search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        if time_left() < 20:
            break

        # Kick
        kicked_rows = kick(best_rows)

        # Local search on kicked rows
        polished_kicked = local_polish(kicked_rows)

        # Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final polish
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity check
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
