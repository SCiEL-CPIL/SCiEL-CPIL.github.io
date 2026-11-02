import random
import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Uses a matching-inspired approach: generate diverse candidates and select
    to minimize the maximum scope while ensuring disjoint difference sets.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer

    def get_scope(rows):
        return max(max(row) for row in rows) if rows else 0

    def compute_row_diffs(row):
        """Compute all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def generate_row(used_diffs, max_val_bound, randomize=True, diversity_factor=1.0):
        """
        Generate a valid row of k+1 integers starting at 0,
        with max entry <= max_val_bound, and differences disjoint from used_diffs.
        """
        row = [0]
        for pos in range(1, k + 1):
            candidates = []
            x = row[-1] + 1
            scan_limit = max_val_bound + 1 if max_val_bound is not None else 200
            
            while x < scan_limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    candidates.append(x)
                    # Collect a few candidates for diversity
                    if len(candidates) >= max(3, int(5 * diversity_factor)):
                        break
                x += 1

            if not candidates:
                # Fallback: scan higher
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                        break
                    x += 1
                if not candidates:
                    return None

            if randomize and len(candidates) > 1:
                # Weighted selection favoring smaller values
                weights = [1.0 / (1.0 + c * 0.02) for c in candidates]
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0.0
                chosen = candidates[-1]
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = candidates[i]
                        break
            else:
                chosen = candidates[0]

            row.append(chosen)
            used_diffs.update({chosen - a for a in row[:-1]})

        return row

    def local_search(rows, used_diffs, time_budget):
        """
        Local search to reduce the scope by replacing rows with smaller valid ones.
        """
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            max_val = max(max(r) for r in rows)

            # Try to improve rows with max value
            for i, row in enumerate(rows):
                if max(row) != max_val:
                    continue

                current_diffs = compute_row_diffs(row)
                available_diffs = used_diffs - current_diffs
                new_max_bound = max_val - 1

                if new_max_bound < k:
                    continue

                # Try multiple random replacements
                for attempt in range(15):
                    temp_used = set(available_diffs)
                    new_row = generate_row(temp_used, new_max_bound, randomize=True,
                                           diversity_factor=1.0 + attempt * 0.3)
                    if new_row is not None and max(new_row) < max_val:
                        used_diffs -= current_diffs
                        new_diffs = compute_row_diffs(new_row)
                        used_diffs |= new_diffs
                        rows[i] = new_row
                        improved = True
                        break

                if time.time() - start_time > time_budget:
                    break

            # Also try to improve near-max rows to free up space
            if not improved and time.time() - start_time < time_budget - 5:
                for i, row in enumerate(rows):
                    if max(row) < max_val - 3:
                        continue

                    current_diffs = compute_row_diffs(row)
                    available_diffs = used_diffs - current_diffs
                    new_max_bound = max(row) - 1

                    if new_max_bound < k:
                        continue

                    for attempt in range(5):
                        temp_used = set(available_diffs)
                        new_row = generate_row(temp_used, new_max_bound, randomize=True,
                                               diversity_factor=1.0 + attempt * 0.5)
                        if new_row is not None and max(new_row) < max(row):
                            used_diffs -= current_diffs
                            new_diffs = compute_row_diffs(new_row)
                            used_diffs |= new_diffs
                            rows[i] = new_row
                            improved = True
                            break

                    if time.time() - start_time > time_budget:
                        break

        return rows, used_diffs

    def solve():
        best_rows = None
        best_scope = float('inf')

        # Phase 1: Initial generation with randomized greedy
        num_initial_tries = 20
        for trial in range(num_initial_tries):
            if time.time() - start_time > time_limit - 40:
                break

            random.seed(trial * 137 + 42)
            used_diffs = set()
            rows = []

            for _ in range(n_rows):
                row = [0]
                for pos in range(1, k + 1):
                    # Collect candidates with diversity
                    candidates = []
                    x = row[-1] + 1
                    while x < 150:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                            if len(candidates) >= 8:
                                break
                        x += 1

                    if not candidates:
                        x = row[-1] + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if not (new_diffs & used_diffs):
                                candidates.append(x)
                                break
                            x += 1

                    # Weighted random selection
                    if len(candidates) > 1:
                        weights = [1.0 / (1.0 + c * 0.015) for c in candidates]
                        total_w = sum(weights)
                        r = random.random() * total_w
                        cumulative = 0.0
                        chosen = candidates[-1]
                        for ci, w in enumerate(weights):
                            cumulative += w
                            if r <= cumulative:
                                chosen = candidates[ci]
                                break
                    else:
                        chosen = candidates[0]

                    row.append(chosen)
                    used_diffs.update({chosen - a for a in row[:-1]})

                rows.append(row)

            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                best_used_diffs = set(used_diffs)

                # Apply local search
                if scope <= 135 and time.time() - start_time < time_limit - 15:
                    improved_rows, improved_diffs = local_search(
                        [r[:] for r in rows], set(used_diffs), time_limit - 5
                    )
                    improved_scope = get_scope(improved_rows)
                    if improved_scope < best_scope:
                        best_scope = improved_scope
                        best_rows = [r[:] for r in improved_rows]
                        best_used_diffs = set(improved_diffs)

            if best_scope <= 113:
                break

        # Phase 2: Diverse restarts with different strategies
        if best_rows is None or time.time() - start_time < time_limit - 20:
            for seed in range(40):
                if time.time() - start_time > time_limit - 10:
                    break

                random.seed(seed * 251 + 17)
                used_diffs = set()
                rows = []

                # Vary the generation strategy per seed
                diversity = 0.5 + (seed % 5) * 0.3
                max_scan = 120 + (seed % 10) * 5

                for _ in range(n_rows):
                    row = [0]
                    for pos in range(1, k + 1):
                        candidates = []
                        x = row[-1] + 1
                        while x < max_scan:
                            new_diffs = {x - a for a in row}
                            if not (new_diffs & used_diffs):
                                candidates.append(x)
                                if len(candidates) >= max(3, int(6 * diversity)):
                                    break
                            x += 1

                        if not candidates:
                            x = row[-1] + 1
                            while True:
                                new_diffs = {x - a for a in row}
                                if not (new_diffs & used_diffs):
                                    candidates.append(x)
                                    break
                                x += 1

                        if len(candidates) > 1:
                            weights = [1.0 / (1.0 + c * 0.02) for c in candidates]
                            total_w = sum(weights)
                            r = random.random() * total_w
                            cumulative = 0.0
                            chosen = candidates[-1]
                            for ci, w in enumerate(weights):
                                cumulative += w
                                if r <= cumulative:
                                    chosen = candidates[ci]
                                    break
                        else:
                            chosen = candidates[0]

                        row.append(chosen)
                        used_diffs.update({chosen - a for a in row[:-1]})

                    rows.append(row)

                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                    best_used_diffs = set(used_diffs)

                    if scope <= 128 and time.time() - start_time < time_limit - 15:
                        improved_rows, improved_diffs = local_search(
                            [r[:] for r in rows], set(used_diffs), time_limit - 5
                        )
                        improved_scope = get_scope(improved_rows)
                        if improved_scope < best_scope:
                            best_scope = improved_scope
                            best_rows = [r[:] for r in improved_rows]
                            best_used_diffs = set(improved_diffs)

                if best_scope <= 112:
                    break

        return best_rows

    return solve()
