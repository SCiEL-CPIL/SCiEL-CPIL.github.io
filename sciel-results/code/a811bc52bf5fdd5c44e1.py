"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit

    def now():
        return time.time()

    def time_left():
        return deadline - now()

    # Each row has k+1 elements
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def diff_set_of_row(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def validate(rows):
        used = set()
        for r in rows:
            if not r or r[0] != 0:
                return False
            for a, b in zip(r, r[1:]):
                if b <= a:
                    return False
            for d in diff_set_of_row(r):
                if d in used:
                    return False
                used.add(d)
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Fast row builder using boolean array for used diffs ----------
    def build_row_greedy(used_arr, scope=None, temp=0.0):
        """Build one row greedily. used_arr is a boolean list/array of used diffs.
        Returns (row, new_used_arr) or (None, used_arr)."""
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            if scope is not None and x > scope:
                return None, used_arr
            new_diffs = [x - a for a in row]
            conflict = False
            for d in new_diffs:
                if d <= len(used_arr) - 1 and used_arr[d]:
                    conflict = True
                    break
            if conflict:
                continue
            if temp > 0 and random.random() < temp and x < 500:
                continue
            row.append(x)
            for d in new_diffs:
                if d >= len(used_arr):
                    used_arr.extend([False] * (d - len(used_arr) + 1))
                used_arr[d] = True
        return row, used_arr

    def build_row_backtrack(used_arr, scope, time_limit=None, rng=None):
        """
        Build one row of size k+1 with max <= scope, all diffs disjoint from used_arr.
        Returns (row, new_used_arr) or (None, None).
        Uses DFS with aggressive pruning and optional randomized candidate ordering.
        """
        if time_limit is not None and now() > time_limit:
            return None, None

        if rng is None:
            rng = random

        row = [0]
        cur_used = list(used_arr)
        target_len = row_len

        # Ensure cur_used is large enough
        needed = scope + 1
        if len(cur_used) < needed:
            cur_used.extend([False] * (needed - len(cur_used)))

        sys.setrecursionlimit(10000)

        def dfs(row, cur_used, depth):
            if time_limit is not None and now() > time_limit:
                return None
            if len(row) == target_len:
                return list(row), list(cur_used)

            last = row[-1]
            rem = target_len - len(row)

            min_next = last + 1
            max_next = scope - (rem - 1)

            if min_next > max_next:
                return None

            # Generate candidates
            candidates = list(range(min_next, max_next + 1))

            # Randomize for diversity
            if rng.random() < 0.3:
                rng.shuffle(candidates)
            else:
                # Try a mix: some small, some random
                if len(candidates) > 5:
                    # Keep first few in order, shuffle the rest
                    split = min(3, len(candidates))
                    first = candidates[:split]
                    rest = candidates[split:]
                    rng.shuffle(rest)
                    candidates = first + rest

            for x in candidates:
                if time_limit is not None and now() > time_limit:
                    return None

                # Check conflicts
                conflict = False
                for a in row:
                    d = x - a
                    if d < len(cur_used) and cur_used[d]:
                        conflict = True
                        break
                if conflict:
                    continue

                # Save state
                old_used = list(cur_used)
                new_diffs = [x - a for a in row]
                for d in new_diffs:
                    if d >= len(cur_used):
                        cur_used.extend([False] * (d - len(cur_used) + 1))
                    cur_used[d] = True

                row.append(x)
                res = dfs(row, cur_used, depth + 1)
                row.pop()

                if res is not None:
                    return res

                # Restore
                cur_used[:] = old_used

            return None

        res = dfs(row, cur_used, 0)
        if res is None:
            return None, None
        return res

    # ---------- Convert between set and array for used diffs ----------
    def set_to_arr(diffs_set, size=None):
        if size is None:
            size = max(diffs_set) + 1 if diffs_set else 1
        arr = [False] * size
        for d in diffs_set:
            if d < size:
                arr[d] = True
        return arr

    def arr_to_set(arr):
        return {i for i, v in enumerate(arr) if v}

    # ---------- Initial construction ----------
    random.seed(42)
    rng = random

    # Pure greedy initial
    used_arr = [False] * 200
    rows = []
    for _ in range(n_rows):
        row, used_arr = build_row_greedy(used_arr, temp=0.0)
        if row is None:
            # Fallback with larger scope
            used_arr = [False] * 500
            row, used_arr = build_row_greedy(used_arr, temp=0.0)
        rows.append(row)

    best_rows = [list(r) for r in rows]
    best_max = max_val(best_rows)

    # ---------- Multi-temperature greedy restarts ----------
    temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    attempts = 0
    max_attempts = 200
    while time_left() > 30 and attempts < max_attempts:
        attempts += 1
        temp = temps[attempts % len(temps)]
        used_arr = [False] * 200
        trial_rows = []
        ok = True
        for _ in range(n_rows):
            if time_left() < 30:
                ok = False
                break
            row, used_arr = build_row_greedy(used_arr, temp=temp)
            if row is None:
                ok = False
                break
            trial_rows.append(row)
        if not ok:
            continue
        mv = max_val(trial_rows)
        if mv < best_max:
            best_max = mv
            best_rows = [list(r) for r in trial_rows]

    # ---------- Backtracking row replacement ----------
    def compute_fixed_used(exclude_idx):
        """Compute the used-diff array for all rows except exclude_idx."""
        used = set()
        for j, r in enumerate(best_rows):
            if j == exclude_idx:
                continue
            used |= diff_set_of_row(r)
        size = max(used) + 1 if used else 1
        size = max(size, best_max + 1)
        return set_to_arr(used, size)

    def try_replace_row(i, scope_target, time_limit=None, rng=None):
        """Try to find a valid replacement for row i with max <= scope_target."""
        if rng is None:
            rng = random
        fixed_used = compute_fixed_used(i)
        # Try multiple scopes from target up to current max
        for scope in range(scope_target, min(best_max, scope_target + 5) + 1):
            if time_limit is not None and now() > time_limit:
                return None
            res, _ = build_row_backtrack(fixed_used, scope, time_limit=time_limit, rng=rng)
            if res is not None:
                # Validate
                new_rows = [r if j != i else res for j, r in enumerate(best_rows)]
                if validate(new_rows):
                    return res
        return None

    # Iterative row replacement to reduce scope
    improved = True
    while improved and time_left() > 25:
        improved = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if time_left() < 25:
                break
            cur_max_row = best_rows[i][-1]
            target = cur_max_row - 1
            if target < k:  # minimum possible max for k+1 elements starting at 0
                continue
            new_row = try_replace_row(i, target, time_limit=deadline - 25, rng=rng)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                mv = max_val(new_rows)
                if mv < best_max:
                    best_max = mv
                    best_rows = new_rows
                    improved = True

    # ---------- Pair-swap refinement ----------
    if time_left() > 35:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        rng.shuffle(pairs)
        for (i, j) in pairs:
            if time_left() < 25:
                break
            other_rows = [r for x, r in enumerate(best_rows) if x != i and x != j]
            fixed_used = set()
            for r in other_rows:
                fixed_used |= diff_set_of_row(r)
            size = max(fixed_used) + 1 if fixed_used else 1
            size = max(size, best_max + 1)
            fixed_arr = set_to_arr(fixed_used, size)

            for scope in range(best_max - 1, max(best_max - 10, k) - 1, -1):
                if time_left() < 25:
                    break
                r1, used_after1 = build_row_backtrack(fixed_arr, scope, time_limit=deadline - 25, rng=rng)
                if r1 is None:
                    continue
                # Extend used_after1 if needed
                needed = scope + 1
                if len(used_after1) < needed:
                    used_after1.extend([False] * (needed - len(used_after1)))
                r2, _ = build_row_backtrack(used_after1, scope, time_limit=deadline - 25, rng=rng)
                if r2 is None:
                    continue
                new_rows = [r if x not in (i, j) else (r1 if x == i else r2)
                            for x, r in enumerate(best_rows)]
                if validate(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                    break

    # ---------- Final polish: aggressive single-row backtracking ----------
    while time_left() > 10:
        any_improve = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if time_left() < 10:
                break
            cur_max_row = best_rows[i][-1]
            # Try multiple scopes below current row max
            for target in range(min(cur_max_row - 1, best_max - 1), max(cur_max_row - 8, k) - 1, -1):
                if time_left() < 10:
                    break
                new_row = try_replace_row(i, target, time_limit=deadline - 10, rng=rng)
                if new_row is not None:
                    new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                        any_improve = True
                        break
        if not any_improve:
            break

    # ---------- Final validation and cleanup ----------
    # Ensure all rows are valid
    for i in range(n_rows):
        r = best_rows[i]
        if not r or r[0] != 0:
            best_rows[i] = list(range(row_len))
        else:
            for a, b in zip(r, r[1:]):
                if b <= a:
                    best_rows[i] = list(range(row_len))
                    break

    return best_rows
# EVOLVE-BLOCK-END
