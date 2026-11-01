"""Difference triangle set solver: ILS with row-rebuild + targeted single-element
local search + ConflictTracker for fast incremental conflict detection."""
import time
import random
import math


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2

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
    class ConflictTracker:
        def __init__(self, rows, size):
            self.conf = [0] * (size + 1)
            self.n_conf = 0
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
        row, x = [0], 0
        row_diffs = set()
        while len(row) < row_len:
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

    # ---------- Targeted single-element local search (from inspiration) ----------
    def targeted_descent(rows, time_limit):
        """Try to reduce individual elements, especially the global max."""
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = scope(best_rows)

        while time.time() < start + time_limit and time_left() > 10:
            # Find global max
            max_val = -1
            max_r = -1
            max_c = -1
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r = r_idx
                        max_c = c_idx

            if max_c == 0:
                break

            current_val = max_val
            prev_val = best_rows[max_r][max_c - 1]
            row = best_rows[max_r]

            # Build fixed diffs (all diffs not involving the target element)
            all_diffs = set()
            for r in best_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        all_diffs.add(r[j] - r[i])

            remove_set = set()
            for i in range(max_c):
                remove_set.add(current_val - row[i])
            for j in range(max_c + 1, len(row)):
                remove_set.add(row[j] - current_val)

            fixed_diffs = all_diffs - remove_set

            found = False
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                for i in range(max_c):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(max_c + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue

                new_left = [new_val - row[i] for i in range(max_c)]
                new_right = [row[j] - new_val for j in range(max_c + 1, len(row))]
                all_new = new_left + new_right
                if len(set(all_new)) != len(all_new):
                    continue

                row[max_c] = new_val
                found = True
                ns = scope(best_rows)
                if ns < best_scope:
                    best_scope = ns
                break

            if not found:
                # Try a random element jolt
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx - 1]
                next_v = best_rows[r_idx][c_idx + 1] if c_idx < row_len - 1 else curr_val + 100
                row = best_rows[r_idx]

                all_diffs = set()
                for r in best_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            all_diffs.add(r[j] - r[i])

                remove_set = set()
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, len(row)):
                    remove_set.add(row[j] - curr_val)
                fixed_diffs = all_diffs - remove_set

                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len - 1 else curr_val + 50
                for _ in range(20):
                    if low > high:
                        break
                    cand = random.randint(low, high)
                    if cand == curr_val:
                        continue
                    valid = True
                    for i in range(c_idx):
                        d = cand - row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    for j in range(c_idx + 1, len(row)):
                        d = row[j] - cand
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_left = [cand - row[i] for i in range(c_idx)]
                    new_right = [row[j] - cand for j in range(c_idx + 1, len(row))]
                    all_new = new_left + new_right
                    if len(set(all_new)) != len(all_new):
                        continue
                    row[c_idx] = cand
                    break

        return best_rows, best_scope

    # ---------- Per-row local optimization ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=80):
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
            jitter = 0.1 if attempt > 10 else 0.0
            while len(row) < row_len:
                found = False
                for trial_x in range(x + 1, x + 150):
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
        while improved and time_left() > 10 and passes < max_passes:
            improved = False
            passes += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 10:
                    break
                used_other = used_set_except(rows, i)
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=70)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    # ---------- Kick: rebuild 1-2 rows ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = used_set_except(rows, idx)
            done = False
            for attempt in range(25):
                row = greedy_row(used_other, jitter=0.35, cap=600)
                if row is not None:
                    rows[idx] = row
                    done = True
                    break
            if not done:
                row = greedy_row(used_other, jitter=0.0, cap=10**6)
                if row is not None:
                    rows[idx] = row
        return rows

    # ---------- Initialization ----------
    best_rows = greedy_full(jitter=0.0)
    best_scope = scope(best_rows)

    if time_left() > 40:
        for r in range(8):
            if time_left() < 40:
                break
            cand = greedy_full(jitter=random.uniform(0.05, 0.3))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    # Initial polish
    if time_left() > 20:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # ---------- ILS loop: targeted descent + row rebuild + polish ----------
    while time_left() > 35:
        # 1. Targeted single-element descent (from inspiration)
        if time_left() > 40:
            desc_rows, desc_scope = targeted_descent(best_rows, time_limit=min(8, time_left() - 35))
            if desc_scope < best_scope:
                best_rows = desc_rows
                best_scope = desc_scope

        # 2. Row-level polish
        if time_left() > 30:
            polished = local_polish(best_rows, max_passes=4)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)

        if time_left() < 30:
            break

        # 3. Kick (rebuild rows)
        kicked = kick(best_rows)

        # 4. Polish the kicked solution
        if time_left() > 25:
            polished_kicked = local_polish(kicked, max_passes=5)
            if is_valid(polished_kicked):
                s = scope(polished_kicked)
                if s < best_scope:
                    best_rows = polished_kicked
                    best_scope = s

            # 5. Targeted descent on kicked
            if time_left() > 25:
                desc_k, desc_sk = targeted_descent(polished_kicked, time_limit=min(6, time_left() - 25))
                if is_valid(desc_k):
                    if desc_sk < best_scope:
                        best_rows = desc_k
                        best_scope = desc_sk

    # ---------- Final SA pass if time remains ----------
    if time_left() > 40:
        rows = [list(r) for r in best_rows]
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
        while time_left() > 18 and t > t_min:
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
                elif random.random() < math.exp(-penalty / max(t, 1e-6)):
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

        if time_left() > 15:
            polished = local_polish(best_rows, max_passes=3)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        best_rows = greedy_full(jitter=0.0)

    return best_rows
