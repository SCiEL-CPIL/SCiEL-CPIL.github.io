"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)

    # Helper to compute all differences in a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Validate a complete solution
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # Greedy baseline
    def greedy_solution(seed=None):
        if seed is not None:
            random.seed(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    best_used = 0
    for row in best_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                best_used |= (1 << (row[j] - row[i]))

    # Simulated Annealing approach
    def simulated_annealing(initial_rows, deadline_t, initial_temp=100.0, cooling_rate=0.999, max_iter=50000):
        rows = [list(r) for r in initial_rows]
        used_diffs = set()
        for row in rows:
            used_diffs |= row_diffs(row)

        current_max = get_max_entry(rows)
        best_rows_sa = [list(r) for r in rows]
        best_max_sa = current_max
        best_used_sa = set(used_diffs)

        temp = initial_temp
        iter_count = 0

        while time.time() < deadline_t and temp > 0.01:
            iter_count += 1

            # Pick a random row and position to modify
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]

            # Strategy 1: Replace an element with a smaller one
            if len(row) > 1 and random.random() < 0.6:
                pos = random.randint(1, len(row) - 1)
                old_val = row[pos]

                # Try to replace with a smaller value
                new_val_candidates = list(range(max(1, old_val - 5), old_val))
                random.shuffle(new_val_candidates)

                for new_val in new_val_candidates[:3]:
                    # Check if new_val is valid (greater than previous, less than next)
                    if pos > 0 and new_val <= row[pos - 1]:
                        continue
                    if pos < len(row) - 1 and new_val >= row[pos + 1]:
                        continue

                    # Compute new differences
                    new_row = row[:]
                    new_row[pos] = new_val

                    # Check internal consistency
                    valid = True
                    new_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            if d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if not valid:
                            break

                    if not valid:
                        continue

                    # Check global conflicts
                    # Remove old diffs
                    old_diffs = row_diffs(row)
                    remaining_diffs = used_diffs - old_diffs

                    if new_diffs & remaining_diffs:
                        continue

                    # Accept or reject based on temperature
                    new_max = get_max_entry(rows) if new_row[-1] < current_max else current_max
                    delta = new_max - current_max

                    if delta < 0 or random.random() < math.exp(-delta / temp):
                        # Accept
                        used_diffs = remaining_diffs | new_diffs
                        rows[row_idx] = new_row
                        current_max = get_max_entry(rows)

                        if current_max < best_max_sa:
                            best_max_sa = current_max
                            best_rows_sa = [list(r) for r in rows]
                            best_used_sa = set(used_diffs)
                        break
            else:
                # Strategy 2: Swap elements between two rows
                if n_rows > 1:
                    row_idx2 = random.randint(0, n_rows - 1)
                    if row_idx2 == row_idx:
                        row_idx2 = (row_idx + 1) % n_rows

                    row2 = rows[row_idx2]

                    if len(row) > 1 and len(row2) > 1:
                        pos1 = random.randint(1, len(row) - 1)
                        pos2 = random.randint(1, len(row2) - 1)

                        val1 = row[pos1]
                        val2 = row2[pos2]

                        if val1 == val2:
                            continue

                        # Create swapped rows
                        new_row1 = row[:]
                        new_row2 = row2[:]
                        new_row1[pos1] = val2
                        new_row2[pos2] = val1

                        # Check if still increasing
                        if any(new_row1[i] >= new_row1[i + 1] for i in range(len(new_row1) - 1)):
                            continue
                        if any(new_row2[i] >= new_row2[i + 1] for i in range(len(new_row2) - 1)):
                            continue

                        # Compute diffs
                        diff1 = row_diffs(new_row1)
                        diff2 = row_diffs(new_row2)

                        # Check internal consistency
                        if len(diff1) != diffs_per_row or len(diff2) != diffs_per_row:
                            continue

                        # Check global conflicts
                        old_diff1 = row_diffs(row)
                        old_diff2 = row_diffs(row2)
                        remaining_diffs = used_diffs - old_diff1 - old_diff2

                        if (diff1 | diff2) & remaining_diffs:
                            continue

                        new_max = get_max_entry([new_row1] + [r for idx, r in enumerate(rows) if idx not in (row_idx, row_idx2)] + [new_row2])
                        delta = new_max - current_max

                        if delta < 0 or random.random() < math.exp(-delta / temp):
                            used_diffs = remaining_diffs | diff1 | diff2
                            rows[row_idx] = new_row1
                            rows[row_idx2] = new_row2
                            current_max = get_max_entry(rows)

                            if current_max < best_max_sa:
                                best_max_sa = current_max
                                best_rows_sa = [list(r) for r in rows]
                                best_used_sa = set(used_diffs)

            # Cool down
            temp *= cooling_rate

        return best_rows_sa, best_max_sa

    # Iterated Local Search
    def iterated_local_search(initial_rows, deadline_t, max_iter=1000):
        rows = [list(r) for r in initial_rows]
        best_rows_ils = [list(r) for r in rows]
        best_max_ils = get_max_entry(rows)

        for _ in range(max_iter):
            if time.time() > deadline_t:
                break

            # Perturb
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]

            if len(row) > 1:
                # Randomly modify one element
                pos = random.randint(1, len(row) - 1)
                delta = random.randint(-10, 10)
                old_val = row[pos]
                new_val = max(1, old_val + delta)

                # Check if valid position
                if pos > 0 and new_val <= row[pos - 1]:
                    new_val = row[pos - 1] + 1
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    new_val = row[pos + 1] - 1

                if new_val != old_val:
                    rows[row_idx][pos] = new_val

            # Local search: try to fix any issues
            improved = True
            while improved and time.time() < deadline_t:
                improved = False
                for idx in range(n_rows):
                    if time.time() > deadline_t:
                        break
                    r = rows[idx]
                    # Try to reduce max entry
                    if len(r) > 1:
                        last_pos = len(r) - 1
                        old_last = r[last_pos]
                        # Try smaller values
                        for new_last in range(max(2, old_last - 5), old_last):
                            if new_last > r[last_pos - 1] if last_pos > 0 else True:
                                r[last_pos] = new_last
                                # Validate
                                valid = True
                                for i in range(len(r) - 1):
                                    if r[i] >= r[i + 1]:
                                        valid = False
                                        break
                                if valid:
                                    # Check global conflicts
                                    all_diffs = set()
                                    for row_check in rows:
                                        for i in range(len(row_check)):
                                            for j in range(i + 1, len(row_check)):
                                                d = row_check[j] - row_check[i]
                                                if d in all_diffs:
                                                    valid = False
                                                    break
                                                all_diffs.add(d)
                                            if not valid:
                                                break
                                        if not valid:
                                            break
                                    if valid:
                                        improved = True
                                        new_max = get_max_entry(rows)
                                        if new_max < best_max_ils:
                                            best_max_ils = new_max
                                            best_rows_ils = [list(r) for r in rows]
                                        break
                                r[last_pos] = old_last
                                if not valid:
                                    break

        return best_rows_ils, best_max_ils

    # Main search loop
    search_deadline = deadline - 10.0

    # Phase 1: Simulated Annealing
    if time.time() < search_deadline:
        sa_deadline = time.time() + min(60.0, search_deadline - time.time())
        try:
            sa_rows, sa_max = simulated_annealing(best_rows, sa_deadline, initial_temp=50.0, cooling_rate=0.9995, max_iter=30000)
            if sa_max < best_max:
                best_max = sa_max
                best_rows = sa_rows
        except Exception:
            pass

    # Phase 2: Iterated Local Search
    if time.time() < search_deadline:
        ils_deadline = time.time() + min(40.0, search_deadline - time.time())
        try:
            ils_rows, ils_max = iterated_local_search(best_rows, ils_deadline, max_iter=500)
            if ils_max < best_max:
                best_max = ils_max
                best_rows = ils_rows
        except Exception:
            pass

    # Phase 3: More SA with different parameters
    if time.time() < deadline - 15:
        for temp0 in [30.0, 20.0, 10.0]:
            if time.time() > deadline - 15:
                break
            sa_deadline = time.time() + min(15.0, deadline - 15 - time.time())
            try:
                sa_rows, sa_max = simulated_annealing(best_rows, sa_deadline, initial_temp=temp0, cooling_rate=0.999, max_iter=20000)
                if sa_max < best_max:
                    best_max = sa_max
                    best_rows = sa_rows
            except Exception:
                pass

    # Final validation
    if not validate(best_rows):
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)

    return best_rows

# EVOLVE-BLOCK-END
