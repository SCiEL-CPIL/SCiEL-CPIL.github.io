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

    # ---------- Multiple-restart randomized greedy initialization ----------
    best_rows = greedy_full(jitter=0.0)
    best_scope = scope(best_rows)
    if time_left() > 30:
        for r in range(6):
            if time_left() < 30:
                break
            cand = greedy_full(jitter=random.uniform(0.05, 0.25))
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

    # ---------- Iterated Local Search: polish -> kick -> polish ----------
    while time_left() > 30:
        # 1. local search on the incumbent
        polished = local_polish(best_rows, max_passes=5)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
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

    # ---------- Cheap final SA pass if time remains ----------
    if time_left() > 35:
        rows = [list(r) for r in best_rows]
        # order by descending scope so index 0 is the binding constraint
        order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
        sa_rows = [rows[i] for i in order]
        size = max(max(r) for r in sa_rows) + 5
        tracker = ConflictTracker(sa_rows, size)
        cur_conf = tracker.n_conf
        cur_scope = max(sa_rows[0])
        best_sa = [list(r) for r in sa_rows]
        best_energy = (cur_conf, cur_scope)

        t = 1.0
        t_min = 0.03
        cooling = 0.995
        while time_left() > 15 and t > t_min:
            if random.random() < 0.5:
                idx = 0
            else:
                idx = random.randrange(n_rows)
            old_row = sa_rows[idx]
            tracker.remove_row(old_row)
            conf_after_remove = tracker.n_conf
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(sa_rows[j])
            if random.random() < 0.6:
                new_row = greedy_row(used_other, jitter=0.0, cap=10**6)
            else:
                new_row = greedy_row(used_other, jitter=0.1, cap=10**6)
            if new_row is None:
                tracker.add_row(old_row)
                t *= cooling
                continue
            tracker.add_row(new_row)
            conf_after_add = tracker.n_conf
            new_scope = max(new_row) if idx == 0 else max(sa_rows[0])
            conf_delta = conf_after_add - conf_after_remove
            scope_delta = new_scope - cur_scope
            accept = False
            if conf_delta < 0 or (conf_delta == 0 and scope_delta < 0):
                accept = True
            else:
                penalty = conf_delta * 1000 + scope_delta
                if penalty < 0:
                    accept = True
                elif random.random() < pow(2.718281828, -penalty / max(t, 1e-6)):
                    accept = True
            if accept:
                sa_rows[idx] = new_row
                cur_conf = conf_after_add
                cur_scope = new_scope
                if (cur_conf, cur_scope) < best_energy:
                    best_energy = (cur_conf, cur_scope)
                    best_sa = [list(r) for r in sa_rows]
            else:
                tracker.remove_row(new_row)
                tracker.add_row(old_row)
            t *= cooling

        if best_energy[0] == 0:
            restored = [None] * n_rows
            for i in range(n_rows):
                restored[order[i]] = best_sa[i]
            if is_valid(restored):
                s = scope(restored)
                if s < best_scope:
                    best_rows = restored
                    best_scope = s

        if time_left() > 12:
            polished = local_polish(best_rows, max_passes=4)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        best_rows = greedy_full(jitter=0.0)

    return best_rows
