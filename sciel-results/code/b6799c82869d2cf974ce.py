"""Combined program: bitmask-based SA with greedy initialization."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

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

    def greedy_construct(scope_limit=None, seed=42):
        """Randomized greedy construction."""
        rng = random.Random(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                max_val = scope_limit if scope_limit is not None else 100000
                # Try candidates in a window first, then expand
                window = 50
                for attempt in range(10):
                    upper = min(last_val + window, max_val)
                    candidates = []
                    for x in range(last_val + 1, upper + 1):
                        valid = True
                        for y in row:
                            d = x - y
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                    if candidates:
                        x = rng.choice(candidates)
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                    window *= 2
                if not found:
                    return None
            rows.append(row)
        return rows

    def simulated_annealing_bitmask(initial_rows, time_limit):
        """
        Simulated annealing using a boolean array for fast difference collision checks.
        """
        rows = [r[:] for r in initial_rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Determine max possible difference value for array sizing
        max_diff = best_scope + 100
        # Use a boolean list as a bitset for fast membership tests
        used = [False] * (max_diff + 1)

        # Fill used array
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_diff:
                        used[d] = True
                    else:
                        max_diff = d
                        used = [False] * (max_diff + 1)
                        for r2 in rows:
                            for i2 in range(len(r2)):
                                for j2 in range(i2 + 1, len(r2)):
                                    d2 = r2[j2] - r2[i2]
                                    used[d2] = True
                        break

        t_end = time.time() + time_limit
        initial_temp = max(1.0, current_scope / 3.0)
        cooling_rate = 0.9998
        steps = 0

        while time.time() < t_end:
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)

            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 30

            if min_val > max_val:
                initial_temp *= cooling_rate
                continue

            # Collect old diffs involving position pos
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.append(d)

            # Try multiple candidate values
            n_attempts = random.randint(1, 5)
            accepted = False
            for _ in range(n_attempts):
                # Generate candidate: sometimes near current, sometimes random
                if random.random() < 0.5:
                    # Local perturbation
                    span = max(1, (max_val - min_val) // 10)
                    new_val = old_val + random.randint(-span, span)
                else:
                    new_val = random.randint(min_val, max_val)

                if new_val < min_val or new_val > max_val or new_val == old_val:
                    continue

                # Remove old diffs from used
                for d in old_diffs:
                    if d <= max_diff:
                        used[d] = False

                # Try adding new diffs
                new_diffs = []
                valid = True
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = new_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - new_val
                        if d > max_diff or used[d]:
                            valid = False
                            break
                        new_diffs.append(d)

                if not valid:
                    # Restore old diffs
                    for d in old_diffs:
                        if d <= max_diff:
                            used[d] = True
                    continue

                # Compute new scope
                new_row = rows[row_idx][:]
                new_row[pos] = new_val
                new_row_max = max(new_row)
                other_max = 0
                for i in range(n_rows):
                    if i != row_idx:
                        if rows[i][-1] > other_max:
                            other_max = rows[i][-1]
                new_scope = max(other_max, new_row_max)

                # Accept or reject with simulated annealing
                delta_scope = new_scope - current_scope
                if delta_scope <= 0 or (initial_temp > 0.1 and random.random() < pow(2.718281828, -delta_scope / initial_temp)):
                    rows[row_idx][pos] = new_val
                    for d in new_diffs:
                        if d <= max_diff:
                            used[d] = True
                    current_scope = new_scope

                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                    accepted = True
                    break
                else:
                    # Restore old diffs
                    for d in old_diffs:
                        if d <= max_diff:
                            used[d] = True

            initial_temp *= cooling_rate
            steps += 1

        return best_rows, best_scope

    def local_search_polish(rows, time_limit):
        """
        Targeted local search: try to reduce the maximum entry by modifying
        the row that achieves it, one element at a time from the end.
        """
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        t_end = time.time() + time_limit

        max_diff = best_scope + 100
        used = [False] * (max_diff + 1)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_diff:
                        used[d] = True

        # Find the row with the maximum last element
        max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])

        while time.time() < t_end:
            row = rows[max_row_idx]
            if row[-1] <= 0:
                break

            # Try to reduce the last element
            old_val = row[-1]
            new_val = old_val - 1
            if new_val <= row[-2]:
                break

            old_diffs = [old_val - row[i] for i in range(k)]

            # Remove old diffs
            for d in old_diffs:
                if d <= max_diff:
                    used[d] = False

            # Check if new_val is valid
            valid = True
            new_diffs = []
            for i in range(k):
                d = new_val - row[i]
                if d > max_diff or used[d]:
                    valid = False
                    break
                new_diffs.append(d)

            if valid:
                row[-1] = new_val
                for d in new_diffs:
                    used[d] = True
                current_scope = max(current_scope - 1, max(rows[i][-1] for i in range(n_rows) if i != max_row_idx))
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Restore old diffs
                for d in old_diffs:
                    if d <= max_diff:
                        used[d] = True
                # Try to reduce an earlier element
                # Find the second-to-last element and try to reduce it
                pos = k - 1
                if pos <= 0:
                    break
                old_val2 = row[pos]
                new_val2 = old_val2 - 1
                if new_val2 <= row[pos - 1]:
                    # Try to reduce the last element more aggressively
                    # Or break if we can't make progress
                    break

                old_diffs2 = []
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = old_val2 - row[i]
                        else:
                            d = row[i] - old_val2
                        old_diffs2.append(d)

                for d in old_diffs2:
                    if d <= max_diff:
                        used[d] = False

                valid2 = True
                new_diffs2 = []
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = new_val2 - row[i]
                        else:
                            d = row[i] - new_val2
                        if d > max_diff or used[d]:
                            valid2 = False
                            break
                        new_diffs2.append(d)

                if valid2:
                    row[pos] = new_val2
                    for d in new_diffs2:
                        used[d] = True
                    # Update scope
                    new_row_max = max(row)
                    other_max = max(rows[i][-1] for i in range(n_rows) if i != max_row_idx)
                    current_scope = max(other_max, new_row_max)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    for d in old_diffs2:
                        if d <= max_diff:
                            used[d] = True
                    break

            # Update max_row_idx
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])

        return best_rows, best_scope

    # --- Main search loop ---
    best_rows = None
    best_max = float('inf')

    # Phase 1: Get a baseline using greedy with multiple seeds
    for seed in range(42, 62):
        if time.time() > deadline:
            break
        rows = greedy_construct(seed=seed)
        if rows is not None:
            row_max = max(max(r) for r in rows)
            if row_max < best_max:
                best_max = row_max
                best_rows = rows

    if best_rows is None:
        # Fallback: simple greedy without randomization
        used_diffs = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last_val = row[-1]
                for x in range(last_val + 1, last_val + 1000):
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
                        break
            best_rows.append(row)
        best_max = max(max(r) for r in best_rows)

    # Phase 2: Simulated annealing with bitmask
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        sa_time = min(remaining_time - 10, 80)
        improved, imp_max = simulated_annealing_bitmask(best_rows, sa_time)
        if imp_max < best_max:
            best_max = imp_max
            best_rows = improved

    # Phase 3: Local search polish
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        polish_time = min(remaining_time - 5, 30)
        polished, pol_max = local_search_polish(best_rows, polish_time)
        if pol_max < best_max:
            best_max = pol_max
            best_rows = polished

    # Phase 4: If time remains, do more SA restarts
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for restart in range(5):
            if time.time() > deadline:
                break
            sa_time = (deadline - time.time()) / (5 - restart) - 2
            if sa_time < 5:
                break
            improved, imp_max = simulated_annealing_bitmask(best_rows, sa_time)
            if imp_max < best_max:
                best_max = imp_max
                best_rows = improved

    return best_rows
# EVOLVE-BLOCK-END
