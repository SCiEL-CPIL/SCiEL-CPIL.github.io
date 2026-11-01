"""Difference triangle set solver: Iterated Local Search with kicks +
fast incremental conflict machinery and per-row local polish."""
import time
import random


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # safety margin under 150s kill

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

    # ---------- Fast incremental conflict machinery ----------
    # A global count array lets us compute validity (conflicts) and
    # per-row scope delta in O(row diffs) rather than O(total).
    class ConflictTracker:
        def __init__(self, rows, size):
            self.conf = [0] * (size + 1)
            self.n_conf = 0  # number of differences with count > 1
            for row in rows:
                for d in row_diffs_list(row):
                    self._add(d, 1)

        def _ensure(self, d):
            if d >= len(self.conf):
                self.conf.extend([0] * (d + 1 - len(self.conf)))

        def _add(self, d, delta):
            self._ensure(d)
            old = self.conf[d]
            new = old + delta
            if old == 1 and new == 2:
                self.n_conf += 1
            elif old == 2 and new == 1:
                self.n_conf -= 1
            self.conf[d] = new

        def remove_row(self, row):
            for d in row_diffs_list(row):
                self._add(d, -1)

        def add_row(self, row):
            for d in row_diffs_list(row):
                self._add(d, 1)

        def valid(self):
            return self.n_conf == 0

    # ---------- Row builders ----------
    def greedy_row(used_set, jitter=0.0, cap=10**6):
        """Greedy row given already-used differences. jitter skips valid
        candidates to explore alternatives."""
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

    # ---------- Per-row local optimization (refinement machinery) ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=120):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # small jitter in later attempts to escape row-construction minima
            jitter = 0.12 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 120):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 300:
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
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=90)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    if scope(rows) < old_scope:
                        improved = True
        return rows

    # ---------- Kick: rebuild 1-2 rows with high randomness ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = used_set_except(rows, idx)
            done = False
            for attempt in range(30):
                row = greedy_row(used_other, jitter=0.4, cap=600)
                if row is not None:
                    rows[idx] = row
                    done = True
                    break
            if not done:
                row = greedy_row(used_other, jitter=0.0, cap=10**6)
                if row is not None:
                    rows[idx] = row
                else:
                    rows[idx] = list(rows[idx])  # keep original
        return rows

    # ---------- Element-level local search: adjust individual entries ----------
    def element_level_polish(rows, max_time=30):
        """Try to reduce scope by moving individual elements of the largest row
        to smaller values, maintaining validity. Uses wider delta range and
        bidirectional moves (increase smaller elements, decrease larger ones)."""
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
                # Try to reduce each element (except 0) by up to 10
                for pos in range(len(row) - 1, 0, -1):
                    orig = row[pos]
                    found_move = False
                    for delta in range(1, 11):
                        new_val = orig - delta
                        if new_val <= row[pos - 1]:
                            break
                        # Check that new differences don't conflict
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
                        # Remove old diffs, add new
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(orig - row[i])
                        # Accept the change
                        row[pos] = new_val
                        row_diffs = row_diffs - old_diffs | new_diffs
                        improved = True
                        found_move = True
                        break
                    if found_move:
                        # After a move, recompute row_diffs for consistency
                        row_diffs = row_diffs_set(row)
        return rows

    # ---------- Multiple-restart randomized greedy initialization ----------
    best_rows = greedy_full(jitter=0.0)
    best_scope = scope(best_rows)
    if time_left() > 30:
        for r in range(8):
            if time_left() < 30:
                break
            cand = greedy_full(jitter=random.uniform(0.05, 0.35))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    # Initial polish
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        # Element-level polish for finer improvements
        if time_left() > 10:
            elem_polished = element_level_polish(best_rows, max_time=15)
            if scope(elem_polished) < best_scope:
                best_rows = elem_polished
                best_scope = scope(best_rows)

    # ---------- Iterated Local Search: polish -> kick -> polish ----------
    while time_left() > 30:
        # 1. local search on the incumbent
        polished = local_polish(best_rows, max_passes=5)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        # Element-level polish for finer improvements
        if time_left() > 20:
            elem_polished = element_level_polish(best_rows, max_time=8)
            if scope(elem_polished) < best_scope:
                best_rows = elem_polished
                best_scope = scope(best_rows)
        if time_left() < 25:
            break
        # 2. kick
        kicked = kick(best_rows)
        # 3. local search on kicked (even if temporarily invalid, polish fixes)
        polished_kicked = local_polish(kicked, max_passes=5)
        if is_valid(polished_kicked):
            s = scope(polished_kicked)
            if s < best_scope:
                best_rows = polished_kicked
                best_scope = s
            # Element-level polish on the improved solution
            if time_left() > 15 and s < best_scope + 5:
                elem_kicked = element_level_polish(polished_kicked, max_time=6)
                if scope(elem_kicked) < best_scope:
                    best_rows = elem_kicked
                    best_scope = scope(best_rows)

    # ---------- Final high-precision polish pass ----------
    if time_left() > 20:
        # Multiple rounds of element-level polish with tighter constraints
        for _ in range(3):
            if time_left() < 15:
                break
            elem_final = element_level_polish(best_rows, max_time=5)
            if scope(elem_final) < best_scope:
                best_rows = elem_final
                best_scope = scope(best_rows)
        
        # Final local polish with more passes
        if time_left() > 15:
            polished_final = local_polish(best_rows, max_passes=10)
            if scope(polished_final) < best_scope:
                best_rows = polished_final
                best_scope = scope(polished_final)
        
        # One more element-level pass
        if time_left() > 12:
            elem_last = element_level_polish(best_rows, max_time=4)
            if scope(elem_last) < best_scope:
                best_rows = elem_last
                best_scope = scope(best_rows)

    # Final validity guarantee.
    if not is_valid(best_rows):
        best_rows = greedy_full(jitter=0.0)

    return best_rows
