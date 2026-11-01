import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_used_diffs(rows):
        """Return set of all positive differences across all rows."""
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def row_diffs(row):
        """Return set of all positive differences within a single row."""
        s = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                s.add(row[j] - row[i])
        return s

    def count_conflicts(rows):
        """Count total duplicate differences (how many times a diff appears more than once)."""
        diff_count = defaultdict(int)
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diff_count[r[j] - r[i]] += 1
        conflicts = 0
        for d, c in diff_count.items():
            if c > 1:
                conflicts += c - 1
        return conflicts

    def greedy_row(used_diffs, row_len, max_val, rand_prob=0.0):
        """
        Generate one valid row (starting at 0, length row_len) that avoids used_diffs.
        Returns the row or None if impossible within max_val.
        """
        row = [0]
        while len(row) < row_len:
            last = row[-1]
            candidates = []
            x = last + 1
            while x <= max_val and len(candidates) < 50:
                valid = True
                for a in row:
                    if (x - a) in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
                x += 1

            if not candidates:
                return None

            if rand_prob > 0 and len(candidates) > 1 and random.random() < rand_prob:
                x = random.choice(candidates)
            else:
                x = candidates[0]

            row.append(x)
            for a in row[:-1]:
                used_diffs.add(x - a)

        return row

    def rebuild_row(rows, row_idx, max_val, rand_prob=0.0, seed=None):
        """
        Rebuild row `row_idx` avoiding all differences from other rows.
        Returns (new_rows, success).
        """
        if seed is not None:
            random.seed(seed)

        # Collect all diffs from other rows
        other_diffs = set()
        for i, r in enumerate(rows):
            if i == row_idx:
                continue
            other_diffs |= row_diffs(r)

        # Try to build a new row
        new_row = greedy_row(set(other_diffs), row_len, max_val, rand_prob)
        if new_row is None:
            return rows, False

        new_rows = [r[:] for r in rows]
        new_rows[row_idx] = new_row
        return new_rows, True

    def full_rebuild(n_rows, k, max_val, rand_prob=0.1, seed=None):
        """
        Build all rows from scratch, one at a time, avoiding prior rows' diffs.
        Returns rows or None.
        """
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for i in range(n_rows):
            row = greedy_row(set(used), row_len, max_val, rand_prob)
            if row is None:
                return None
            rows.append(row)
            used |= row_diffs(row)
        return rows

    def pair_conflict_matrix(rows):
        """
        For each pair of rows (i, j), count how many differences are shared.
        Returns a list of (count, i, j) sorted descending.
        """
        row_d = [row_diffs(r) for r in rows]
        conflicts = []
        for i in range(n_rows):
            for j in range(i+1, n_rows):
                shared = row_d[i] & row_d[j]
                if shared:
                    conflicts.append((len(shared), i, j))
        conflicts.sort(reverse=True)
        return conflicts

    # Phase 1: Multi-start greedy with different random seeds
    best_rows = None
    best_scope = float('inf')

    for attempt in range(30):
        if time.time() > deadline - 60:
            break
        rows = full_rebuild(n_rows, k, 200, rand_prob=0.2, seed=attempt)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Fallback: very permissive greedy
        used = set()
        best_rows = []
        for i in range(n_rows):
            row = [0]
            x = 1
            while len(row) < row_len:
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                x += 1
            best_rows.append(row)
        best_scope = get_scope(best_rows)

    # Phase 2: Conflict-driven row reconstruction
    # Repeatedly find the worst-conflicting row pair and rebuild one of them
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline - 30 and no_improve_count < max_no_improve:
        # Find worst conflict pair
        conflicts = pair_conflict_matrix(current_rows)
        if not conflicts:
            break

        worst_count, ri, rj = conflicts[0]

        # Try to rebuild the row that appears in more conflicts
        # Count how many conflicts each row is involved in
        row_conflict_count = [0] * n_rows
        for c, i, j in conflicts:
            row_conflict_count[i] += c
            row_conflict_count[j] += c

        # Rebuild the row with highest conflict involvement
        target_row = ri
        if row_conflict_count[rj] > row_conflict_count[ri]:
            target_row = rj

        # Try multiple rebuild attempts with increasing randomness
        success = False
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            rand_p = 0.1 + 0.05 * attempt
            new_rows, ok = rebuild_row(current_rows, target_row, current_scope, rand_p, seed=attempt * 1000 + ri)
            if ok:
                new_scope = get_scope(new_rows)
                new_conflicts = count_conflicts(new_rows)
                old_conflicts = count_conflicts(current_rows)
                if new_scope <= current_scope and new_conflicts < old_conflicts:
                    current_rows = new_rows
                    current_scope = min(current_scope, new_scope)
                    success = True
                    no_improve_count = 0
                    break
                elif new_scope < current_scope:
                    current_rows = new_rows
                    current_scope = new_scope
                    success = True
                    no_improve_count = 0
                    break
        if not success:
            no_improve_count += 1

    # Phase 3: Simulated annealing on row-level moves
    # Occasionally rebuild a random row to escape local minima
    if time.time() < deadline - 20:
        temp = 10.0
        cooling = 0.95
        for step in range(5000):
            if time.time() > deadline - 20:
                break
            if temp < 0.1:
                break

            # Pick a random row to rebuild
            r_idx = random.randint(0, n_rows - 1)
            new_rows, ok = rebuild_row(current_rows, r_idx, current_scope + 10, rand_prob=0.5, seed=step)
            if not ok:
                temp *= cooling
                continue

            new_scope = get_scope(new_rows)
            delta = new_scope - current_scope

            if delta <= 0 or random.random() < math.exp(-delta / temp):
                current_rows = new_rows
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]

            temp *= cooling

    # Phase 4: Targeted element-level descent on the current best
    # Try to reduce the scope by moving individual elements
    if time.time() < deadline - 5:
        work_rows = [r[:] for r in best_rows]
        work_scope = best_scope
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Find global max
            max_val = -1
            max_r = -1
            max_c = -1
            for ri, r in enumerate(work_rows):
                for ci, v in enumerate(r):
                    if v > max_val:
                        max_val = v
                        max_r = ri
                        max_c = ci

            if max_c == 0:
                break

            prev_v = work_rows[max_r][max_c - 1]
            row = work_rows[max_r]

            # Compute all diffs except those involving (max_r, max_c)
            all_diffs = set()
            for ri, r in enumerate(work_rows):
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        if ri == max_r and (i == max_c or j == max_c):
                            continue
                        all_diffs.add(r[j] - r[i])

            # Try to find a smaller value for position (max_r, max_c)
            found = False
            for new_val in range(max_val - 1, prev_v, -1):
                valid = True
                for i in range(max_c):
                    d = new_val - row[i]
                    if d in all_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(max_c + 1, len(row)):
                    d = row[j] - new_val
                    if d in all_diffs:
                        valid = False
                        break
                if not valid:
                    continue

                # Check internal consistency
                new_left = [new_val - row[i] for i in range(max_c)]
                new_right = [row[j] - new_val for j in range(max_c+1, len(row))]
                if len(set(new_left + new_right)) != len(new_left + new_right):
                    continue

                row[max_c] = new_val
                found = True
                break

            if found:
                new_scope = get_scope(work_rows)
                if new_scope < work_scope:
                    work_scope = new_scope
                    if work_scope < best_scope:
                        best_scope = work_scope
                        best_rows = [r[:] for r in work_rows]
                improved = True

    # Final validation
    if not validate(best_rows):
        # Emergency fallback
        used = set()
        best_rows = []
        for i in range(n_rows):
            row = [0]
            x = 1
            while len(row) < row_len:
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                x += 1
            best_rows.append(row)

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
