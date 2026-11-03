import time
import random
import math
import sys
from collections import defaultdict

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    def greedy_construct():
        """Fast greedy construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    class SAState:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.row_max = [0] * self.n_rows
            self._rebuild()

        def _rebuild(self):
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.row_max = [0] * self.n_rows
            for i, r in enumerate(self.rows):
                self.row_max[i] = r[-1]
                if r[-1] > self.max_val:
                    self.max_val = r[-1]
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        d = r[b] - r[a]
                        if d in self.diff_count:
                            self.diff_count[d] += 1
                            if self.diff_count[d] == 2:
                                self.collisions += 1
                        else:
                            self.diff_count[d] = 1

        def get_cost(self):
            return self.collisions, self.max_val

        def compute_move_delta(self, r_idx, i_idx, new_val):
            """Compute (delta_collisions, delta_max) without applying. Returns None if invalid."""
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return None

            # Check bounds
            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx - 1]:
                return None
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx + 1]:
                return None

            old_row = self.rows[r_idx]

            # Compute affected differences
            changes = {}
            for j in range(self.row_len):
                if j != i_idx:
                    d_old = abs(old_val - old_row[j])
                    d_new = abs(new_val - old_row[j])
                    if d_old != d_new:
                        changes[d_old] = changes.get(d_old, 0) - 1
                        changes[d_new] = changes.get(d_new, 0) + 1

            # If no changes in differences, collision delta is 0
            delta_collisions = 0
            for d, delta in changes.items():
                cur = self.diff_count.get(d, 0)
                new_cnt = cur + delta
                if new_cnt < 0:
                    return None
                old_contrib = max(0, cur - 1)
                new_contrib = max(0, new_cnt - 1)
                delta_collisions += (new_contrib - old_contrib)

            # Compute delta_max
            old_row_max = self.row_max[r_idx]
            new_row_max = max(new_val, old_row_max)
            if i_idx == self.row_len - 1:
                new_row_max = new_val  # last element changed
            else:
                # new_val is in the middle, row max might be old_row_max or new_val
                new_row_max = max(old_row_max, new_val)

            if new_row_max > self.max_val:
                delta_max = new_row_max - self.max_val
            elif old_row_max == self.max_val and new_row_max < self.max_val:
                # Need to find new global max
                other_max = 0
                for i in range(self.n_rows):
                    if i != r_idx:
                        if self.row_max[i] > other_max:
                            other_max = self.row_max[i]
                new_global_max = max(new_row_max, other_max)
                delta_max = new_global_max - self.max_val
            else:
                delta_max = 0

            return delta_collisions, delta_max

        def apply_move(self, r_idx, i_idx, new_val):
            """Apply move and update state. Assumes compute_move_delta returned valid."""
            old_val = self.rows[r_idx][i_idx]
            old_row = self.rows[r_idx]

            # Update diff_count
            for j in range(self.row_len):
                if j != i_idx:
                    d_old = abs(old_val - old_row[j])
                    d_new = abs(new_val - old_row[j])
                    if d_old != d_new:
                        # Decrement old
                        self.diff_count[d_old] -= 1
                        if self.diff_count[d_old] == 1:
                            self.collisions -= 1
                        if self.diff_count[d_old] == 0:
                            del self.diff_count[d_old]
                        # Increment new
                        if d_new in self.diff_count:
                            self.diff_count[d_new] += 1
                            if self.diff_count[d_new] == 2:
                                self.collisions += 1
                        else:
                            self.diff_count[d_new] = 1

            # Apply change
            self.rows[r_idx][i_idx] = new_val

            # Update row_max and global max_val
            old_row_max = self.row_max[r_idx]
            if i_idx == self.row_len - 1:
                self.row_max[r_idx] = new_val
            else:
                self.row_max[r_idx] = max(old_row_max, new_val)

            if self.row_max[r_idx] > self.max_val:
                self.max_val = self.row_max[r_idx]
            elif old_row_max == self.max_val and self.row_max[r_idx] < self.max_val:
                new_max = 0
                for i in range(self.n_rows):
                    if self.row_max[i] > new_max:
                        new_max = self.row_max[i]
                self.max_val = new_max

        def undo_move(self, r_idx, i_idx, old_val):
            """Undo move: restore old_val at (r_idx, i_idx)."""
            new_val = self.rows[r_idx][i_idx]
            if new_val == old_val:
                return

            old_row = self.rows[r_idx]

            # Update diff_count: remove new, add old
            for j in range(self.row_len):
                if j != i_idx:
                    d_new = abs(new_val - old_row[j])
                    d_old = abs(old_val - old_row[j])
                    if d_new != d_old:
                        # Decrement new
                        self.diff_count[d_new] -= 1
                        if self.diff_count[d_new] == 1:
                            self.collisions -= 1
                        if self.diff_count[d_new] == 0:
                            del self.diff_count[d_new]
                        # Increment old
                        if d_old in self.diff_count:
                            self.diff_count[d_old] += 1
                            if self.diff_count[d_old] == 2:
                                self.collisions += 1
                        else:
                            self.diff_count[d_old] = 1

            # Restore value
            self.rows[r_idx][i_idx] = old_val

            # Update row_max and global max_val
            if i_idx == self.row_len - 1:
                self.row_max[r_idx] = old_val
            else:
                self.row_max[r_idx] = max(old_val, max(self.rows[r_idx][0:i_idx], self.rows[r_idx][i_idx+1:]) if i_idx < self.row_len - 1 else old_val)

            if self.row_max[r_idx] > self.max_val:
                self.max_val = self.row_max[r_idx]
            elif new_val == self.max_val and old_val < new_val:
                new_max = 0
                for i in range(self.n_rows):
                    if self.row_max[i] > new_max:
                        new_max = self.row_max[i]
                self.max_val = new_max

    def simulate_annealing(initial_rows, time_limit):
        state = SAState(initial_rows)
        current_collisions, current_max = state.get_cost()
        best_collisions = current_collisions
        best_max = current_max
        best_rows = [r[:] for r in state.rows]

        T = 20.0
        cooling = 0.9995
        start = time.time()
        no_improve = 0
        max_no_improve = 8000

        while time.time() - start < time_limit:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)

            lower = state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = state.rows[r_idx][i_idx + 1] - 1
            else:
                upper = best_max + 8

            if lower > upper:
                continue

            old_val = state.rows[r_idx][i_idx]
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue

            delta = state.compute_move_delta(r_idx, i_idx, new_val)
            if delta is None:
                continue

            d_coll, d_max = delta
            new_collisions = current_collisions + d_coll
            new_max = current_max + d_max

            # Acceptance
            accept = False
            if new_collisions < current_collisions:
                accept = True
            elif new_collisions == current_collisions:
                if new_max <= current_max:
                    accept = True
                else:
                    p = math.exp(-d_max / max(T, 0.5))
                    accept = random.random() < p
            else:
                # More collisions: rarely accept
                penalty = d_coll * 30 + max(0, d_max)
                if penalty < 0:
                    accept = True
                else:
                    p = math.exp(-penalty / max(T, 0.5))
                    accept = random.random() < p and random.random() < 0.05

            if accept:
                state.apply_move(r_idx, i_idx, new_val)
                current_collisions = new_collisions
                current_max = new_max
                if current_collisions < best_collisions or \
                   (current_collisions == best_collisions and current_max < best_max):
                    best_collisions = current_collisions
                    best_max = current_max
                    best_rows = [r[:] for r in state.rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            T *= cooling
            if T < 0.5:
                T = 0.5

            if no_improve > max_no_improve:
                T = 20.0
                no_improve = 0
                # Perturb
                for _ in range(5):
                    rr = random.randint(0, n_rows - 1)
                    ii = random.randint(1, row_len - 1)
                    lo = state.rows[rr][ii - 1] + 1
                    hi = state.rows[rr][ii + 1] - 1 if ii < row_len - 1 else best_max + 10
                    if lo <= hi:
                        nv = random.randint(lo, hi)
                        ov = state.rows[rr][ii]
                        if nv != ov:
                            d = state.compute_move_delta(rr, ii, nv)
                            if d is not None:
                                state.apply_move(rr, ii, nv)
                                current_collisions, current_max = state.get_cost()

        return best_rows, best_max, best_collisions

    def local_polish(rows, time_limit):
        """Aggressive local search to reduce max while maintaining validity."""
        state = SAState(rows)
        best_max = state.max_val
        best_rows = [r[:] for r in state.rows]
        start = time.time()

        while time.time() - start < time_limit:
            if state.collisions > 0:
                # Fix collisions first
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                lower = state.rows[r_idx][i_idx - 1] + 1
                upper = state.rows[r_idx][i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 10
                if lower > upper:
                    continue
                new_val = random.randint(lower, upper)
                old_val = state.rows[r_idx][i_idx]
                if new_val == old_val:
                    continue
                delta = state.compute_move_delta(r_idx, i_idx, new_val)
                if delta is None:
                    continue
                state.apply_move(r_idx, i_idx, new_val)
                continue

            # Valid: try to reduce max
            # Target the row with the largest last element
            max_row_idx = max(range(n_rows), key=lambda i: state.row_max[i])

            # Try reducing the last element of the max row
            i_idx = row_len - 1
            r_idx = max_row_idx
            old_val = state.rows[r_idx][i_idx]
            lower = state.rows[r_idx][i_idx - 1] + 1
            upper = old_val - 1

            if lower <= upper:
                # Try multiple reduction amounts
                for attempt in range(5):
                    if attempt == 0:
                        new_val = old_val - 1
                    elif attempt == 1:
                        new_val = random.randint(lower, max(lower, old_val - 3))
                    else:
                        new_val = random.randint(lower, upper)

                    if new_val == old_val:
                        continue

                    delta = state.compute_move_delta(r_idx, i_idx, new_val)
                    if delta is None:
                        continue

                    d_coll, d_max = delta
                    if d_coll == 0 and d_max < 0:
                        state.apply_move(r_idx, i_idx, new_val)
                        best_max = state.max_val
                        best_rows = [r[:] for r in state.rows]
                        break
                    else:
                        # Try to fix: if this creates collisions, undo and try different
                        state.apply_move(r_idx, i_idx, new_val)
                        # Check if we can fix the collision by moving another element
                        fixed = False
                        for _ in range(3):
                            rr = random.randint(0, n_rows - 1)
                            ii = random.randint(1, row_len - 1)
                            lo = state.rows[rr][ii - 1] + 1
                            hi = state.rows[rr][ii + 1] - 1 if ii < row_len - 1 else best_max + 5
                            if lo <= hi:
                                nv = random.randint(lo, hi)
                                ov = state.rows[rr][ii]
                                if nv != ov:
                                    d2 = state.compute_move_delta(rr, ii, nv)
                                    if d2 is not None and d2[0] < 0:
                                        state.apply_move(rr, ii, nv)
                                        fixed = True
                                        break
                        if not fixed:
                            state.undo_move(rr, ii, ov)
                        # Check final state
                        if state.collisions == 0 and state.max_val < best_max:
                            best_max = state.max_val
                            best_rows = [r[:] for r in state.rows]
                        else:
                            # Revert all changes
                            state._rebuild()

            # Also try reducing non-last elements of the max row
            if time.time() - start < time_limit - 0.5:
                for i_idx in range(1, row_len - 1):
                    r_idx = max_row_idx
                    old_val = state.rows[r_idx][i_idx]
                    lower = state.rows[r_idx][i_idx - 1] + 1
                    upper = state.rows[r_idx][i_idx + 1] - 1
                    if lower > upper:
                        continue
                    for _ in range(2):
                        new_val = random.randint(lower, upper)
                        if new_val == old_val:
                            continue
                        delta = state.compute_move_delta(r_idx, i_idx, new_val)
                        if delta is None:
                            continue
                        d_coll, d_max = delta
                        if d_coll == 0:
                            state.apply_move(r_idx, i_idx, new_val)
                            if state.max_val < best_max:
                                best_max = state.max_val
                                best_rows = [r[:] for r in state.rows]
                            else:
                                state.undo_move(r_idx, i_idx, old_val)
                        else:
                            state.undo_move(r_idx, i_idx, old_val)

        return best_rows, best_max

    # Main
    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(max(r) for r in best_rows)

    # Phase 1: SA to get valid solution
    sa_time1 = min(20.0, deadline - start_time - 5.0)
    if sa_time1 > 2.0:
        local_rows, local_max, local_coll = simulate_annealing(baseline_rows, sa_time1)
        if local_coll == 0 and local_max < best_max:
            best_rows = local_rows
            best_max = local_max

    # Phase 2: Polish
    remaining = deadline - time.time() - 2.0
    if remaining > 1.0:
        polished, polished_max = local_polish(best_rows, remaining)
        if polished_max < best_max:
            best_rows = polished
            best_max = polished_max

    # Phase 3: More SA restarts if time
    remaining = deadline - time.time() - 1.0
    if remaining > 3.0 and best_max > 105:
        while time.time() < deadline - 1.0:
            sa_t = min(5.0, deadline - time.time() - 1.0)
            if sa_t < 1.0:
                break
            init = [r[:] for r in best_rows]
            lr, lm, lc = simulate_annealing(init, sa_t)
            if lc == 0 and lm < best_max:
                best_rows = lr
                best_max = lm
            # Quick polish
            if time.time() < deadline - 1.0:
                pr, pm = local_polish(best_rows, min(2.0, deadline - time.time() - 1.0))
                if pm < best_max:
                    best_rows = pr
                    best_max = pm

    return best_rows
