import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s kill

    total_diffs = n_rows * (k + 1) * k // 2

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_construct(scope_limit, rng=None):
        """Greedy: pick smallest valid value at each step."""
        if rng is None:
            rng = random.Random(42)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                for x in range(last_val + 1, scope_limit + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def greedy_pressure(scope_limit, rng=None):
        """Pressure greedy: minimize the maximum new difference at each step."""
        if rng is None:
            rng = random.Random(42)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                best_x = None
                best_max_diff = float('inf')
                last_val = row[-1]
                for x in range(last_val + 1, scope_limit + 1):
                    max_d = 0
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        if d > max_d:
                            max_d = d
                    if valid and max_d < best_max_diff:
                        best_max_diff = max_d
                        best_x = x
                        # Early exit if we can't do better
                        if max_d == x - last_val:
                            break
                if best_x is None:
                    return None
                row.append(best_x)
                for y in row[:-1]:
                    used_diffs.add(best_x - y)
            rows.append(row)
        return rows

    def rebuild_row(used_diffs, scope_limit, rng):
        """Rebuild a single row greedily, choosing smallest valid values."""
        row = [0]
        while len(row) < k + 1:
            found = False
            last_val = row[-1]
            for x in range(last_val + 1, scope_limit + 1):
                valid = True
                for y in row:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for y in row[:-1]:
                        used_diffs.add(x - y)
                    found = True
                    break
            if not found:
                # Roll back
                for y in row:
                    for z in row:
                        if z > y:
                            used_diffs.discard(z - y)
                return None
        return row

    def row_rebuild_search(rows, time_budget, rng=None):
        """Rebuild-based local search: randomly reconstruct rows to escape local minima."""
        if rng is None:
            rng = random.Random(42)
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])

        t_end = time.time() + time_budget
        no_improve = 0
        max_no_improve = 3000

        while time.time() < t_end and no_improve < max_no_improve:
            row_idx = rng.randint(0, n_rows - 1)
            old_row = rows[row_idx][:]

            # Remove old row's diffs
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    used_diffs.discard(old_row[j] - old_row[i])

            # Try to rebuild with a tighter scope
            target_scope = best_scope
            # Occasionally try slightly looser to find diverse solutions
            if rng.random() < 0.15:
                target_scope = best_scope + rng.randint(1, 10)

            new_row = rebuild_row(used_diffs, target_scope, rng)

            if new_row is None:
                # Restore old
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        used_diffs.add(old_row[j] - old_row[i])
                no_improve += 1
            else:
                rows[row_idx] = new_row
                new_scope = get_scope(rows)

                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    # With some probability, keep the new row even if scope didn't improve
                    # (diversity to escape local minima)
                    if rng.random() < 0.3 and new_scope <= best_scope:
                        no_improve = max(0, no_improve - 1)

        return best_rows, best_scope

    def simulated_annealing(rows, time_budget, rng=None):
        """Simulated annealing to reduce scope."""
        if rng is None:
            rng = random.Random(42)
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])

        t_end = time.time() + time_budget
        initial_temp = max(1.0, current_scope / 3.0)
        current_temp = initial_temp
        cooling_rate = 0.9995
        no_improve = 0
        max_no_improve = 5000

        while time.time() < t_end and no_improve < max_no_improve:
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)

            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 10

            if min_val > max_val:
                current_temp *= cooling_rate
                continue

            # Generate candidate
            if pos == k:
                # Bias toward smaller values
                new_val = rng.randint(min_val, max(min_val + 20, max_val))
            else:
                new_val = rng.randint(min_val, max_val)

            if new_val == old_val:
                current_temp *= cooling_rate
                continue

            # Remove old diffs
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.append(d)
            for d in old_diffs:
                used_diffs.discard(d)

            # Check new diffs
            new_diffs = []
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
            if not valid or len(set(new_diffs)) != len(new_diffs):
                for d in old_diffs:
                    used_diffs.add(d)
                current_temp *= cooling_rate
                continue

            # Compute new scope
            if pos == k:
                new_row_max = new_val
            elif pos > 0:
                new_row_max = max(rows[row_idx][pos + 1:]) if pos < k else new_val
            else:
                new_row_max = max(rows[row_idx])

            other_max = 0
            for i in range(n_rows):
                if i != row_idx:
                    other_max = max(other_max, max(rows[i]))
                else:
                    if pos == k:
                        other_max = max(other_max, max(rows[i][:k]))
                    elif pos < k:
                        other_max = max(other_max, max(rows[i][:pos] + rows[i][pos+1:]))

            new_scope = max(new_row_max, other_max)
            delta = current_scope - new_scope  # positive = improvement

            # Acceptance
            if delta > 0 or rng.random() < math.exp(delta / current_temp):
                for d in new_diffs:
                    used_diffs.add(d)
                rows[row_idx][pos] = new_val
                current_scope = new_scope

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                for d in old_diffs:
                    used_diffs.add(d)
                no_improve += 1

            current_temp *= cooling_rate
            if current_temp < 0.01:
                current_temp = 0.01

        return best_rows, best_scope

    def backtracking_with_scope(scope_limit, time_budget, rng=None):
        """Backtracking to construct rows within a scope limit."""
        if rng is None:
            rng = random.Random(42)
        used_diffs = set()
        rows = []

        def try_build_row(current_row, time_limit):
            if time.time() > time_limit:
                return False

            if len(current_row) == k + 1:
                rows.append(current_row[:])
                return True

            last_val = current_row[-1]
            min_val = last_val + 1
            max_val = min(scope_limit, last_val + scope_limit)

            if min_val > max_val:
                return False

            # Try values, biased toward smaller ones
            candidates = list(range(min_val, max_val + 1))
            if len(candidates) > 30:
                # Mix: some small, some random
                small = candidates[:15]
                rest = candidates[15:]
                rand_sample = rng.sample(rest, min(20, len(rest)))
                candidates = small + rand_sample
                rng.shuffle(candidates[15:])
            else:
                rng.shuffle(candidates)

            for val in candidates:
                if time.time() > time_limit:
                    return False

                new_diffs = []
                valid = True
                for y in current_row:
                    d = val - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if not valid:
                    continue

                # Forward check
                remaining = k + 1 - len(current_row) - 1
                if remaining > 0 and val + remaining > scope_limit:
                    continue

                current_row.append(val)
                for d in new_diffs:
                    used_diffs.add(d)

                if try_build_row(current_row, time_limit):
                    return True

                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)

            return False

        time_limit = time.time() + time_budget
        for _ in range(n_rows):
            row = [0]
            if not try_build_row(row, time_limit):
                return None

        return rows

    # === Main strategy ===

    # Step 1: Get an initial solution using multiple greedy strategies
    initial_rows = None
    best_initial_scope = float('inf')

    strategies = [
        lambda s: greedy_construct(s, random.Random(42)),
        lambda s: greedy_pressure(s, random.Random(42)),
        lambda s: greedy_construct(s, random.Random(123)),
        lambda s: greedy_pressure(s, random.Random(123)),
        lambda s: greedy_construct(s, random.Random(777)),
        lambda s: greedy_pressure(s, random.Random(777)),
    ]

    for scope in [130, 135, 140, 150, 160, 180, 200, 300]:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        for strat in strategies:
            if time.time() > deadline - 5:
                break
            rows = strat(scope)
            if rows is not None and is_valid(rows):
                sc = get_scope(rows)
                if sc < best_initial_scope:
                    best_initial_scope = sc
                    initial_rows = rows

    if initial_rows is None:
        # Fallback: very loose greedy
        initial_rows = greedy_construct(2000, random.Random(42))
        if initial_rows is None:
            # Absolute fallback
            used_diffs = set()
            initial_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for y in row:
                        if x - y in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                initial_rows.append(row)
            return initial_rows

    current_scope = get_scope(initial_rows)

    # Step 2: Row-rebuild search (primary optimizer - escapes local minima)
    remaining = deadline - time.time()
    if remaining > 20:
        rr_rows, rr_scope = row_rebuild_search(initial_rows, remaining * 0.45, random.Random(42))
        if rr_rows is not None and is_valid(rr_rows) and rr_scope < current_scope:
            initial_rows = rr_rows
            current_scope = rr_scope

    # Step 3: More row-rebuild with different seed
    remaining = deadline - time.time()
    if remaining > 15:
        rr2_rows, rr2_scope = row_rebuild_search(initial_rows, remaining * 0.35, random.Random(777))
        if rr2_rows is not None and is_valid(rr2_rows) and rr2_scope < current_scope:
            initial_rows = rr2_rows
            current_scope = rr2_scope

    # Step 4: Simulated annealing for fine-tuning
    remaining = deadline - time.time()
    if remaining > 15:
        sa_rows, sa_scope = simulated_annealing(initial_rows, remaining * 0.3, random.Random(123))
        if sa_rows is not None and is_valid(sa_rows) and sa_scope < current_scope:
            initial_rows = sa_rows
            current_scope = sa_scope

    # Step 5: Final row-rebuild polish
    remaining = deadline - time.time()
    if remaining > 10:
        final_rows, final_scope = row_rebuild_search(initial_rows, remaining * 0.7, random.Random(999))
        if final_rows is not None and is_valid(final_rows) and final_scope < current_scope:
            initial_rows = final_rows
            current_scope = final_scope

    return initial_rows
