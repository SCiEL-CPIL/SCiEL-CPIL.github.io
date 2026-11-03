"""Difference triangle set (7,5) — improved greedy with backtracking + local search."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Greedy construction (same as seed but slightly smarter: try to minimize max)
    def greedy_rows(n_rows, k, randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        if randomize:
            # Perturb: for each row, try shifting some marks up to reduce collisions
            # (this is a rough randomization; real improvement comes from local search)
            pass
        return rows

    def row_diffs(row):
        return frozenset(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))

    def max_entry(rows):
        return max(max(r) for r in rows)

    def is_valid(rows, k):
        all_diffs = []
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    if r[j] <= r[i]:
                        return False
                    all_diffs.append(r[j] - r[i])
        return len(all_diffs) == len(set(all_diffs))

    # Initial greedy solution
    best_rows = greedy_rows(n_rows, k)
    best_score = max_entry(best_rows)

    # Local search: try to reduce max entry by moving the largest mark down
    # and fixing collisions via backtracking on affected rows
    def try_improve(rows, k, time_limit=5.0):
        """Attempt to reduce max entry by local modifications."""
        nonlocal best_rows, best_score, start_time, deadline
        t0 = time.time()
        rows = [list(r) for r in rows]
        current_max = max_entry(rows)
        improved = False

        # Collect all differences
        def all_diffs_set(rows):
            s = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        s.add(r[j] - r[i])
            return s

        # Try to reduce the maximum entry
        # Find the row with the maximum entry
        for attempt in range(200):
            if time.time() > deadline - 20.0 or time.time() - t0 > time_limit:
                break
            if not is_valid(rows, k):
                break

            # Find row with max entry
            max_row_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            target = rows[max_row_idx][-1]

            # Try reducing the last element
            found = False
            for new_last in range(target - 1, 0, -1):
                if time.time() > deadline - 20.0:
                    break
                # Check if new_last creates duplicates within the row
                row = rows[max_row_idx]
                if new_last <= row[-2]:
                    continue
                # Check differences within row
                new_row_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        val = (new_last if j == len(row) - 1 else row[j]) - row[i]
                        if j == len(row) - 1:
                            val = new_last - row[i]
                        new_row_diffs.add(val)
                # Remove old diffs of last element
                old_last_diffs = set(target - row[i] for i in range(len(row) - 1))
                # New diffs
                new_last_diffs = set(new_last - row[i] for i in range(len(row) - 1))
                # All other diffs
                other_diffs = set()
                for i in range(len(row) - 1):
                    for j in range(i + 1, len(row) - 1):
                        other_diffs.add(row[j] - row[i])
                # Check no collision
                if new_last_diffs & other_diffs:
                    continue
                if new_last_diffs & (all_diffs_set(rows) - old_last_diffs):
                    continue
                # Valid! Apply
                rows[max_row_idx][-1] = new_last
                current_max = max_entry(rows)
                improved = True
                found = True
                break
            if not found:
                # Try reducing a middle element
                # Pick a random row and try to reduce a non-last element
                ri = random.randint(0, len(rows) - 1)
                row = rows[ri]
                # Try to reduce each element (except first=0 and last)
                for pos in range(len(row) - 2, 0, -1):
                    if time.time() > deadline - 20.0:
                        break
                    old_val = row[pos]
                    # Try new values from old_val-1 down to row[pos-1]+1
                    for new_val in range(old_val - 1, row[pos - 1], -1):
                        # Check within-row uniqueness
                        new_row = row[:pos] + [new_val] + list(row[pos+1:])
                        # Check strictly increasing
                        if any(new_row[j] <= new_row[j-1] for j in range(1, len(new_row))):
                            continue
                        # Check all diffs
                        new_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                new_diffs.add(new_row[j] - new_row[i])
                        old_diffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                old_diffs.add(row[j] - row[i])
                        # Other rows' diffs
                        other_all = set()
                        for ii in range(len(rows)):
                            if ii == ri:
                                continue
                            for i in range(len(rows[ii])):
                                for j in range(i + 1, len(rows[ii])):
                                    other_all.add(rows[ii][j] - rows[ii][i])
                        # New diffs not in old_diffs must not be in other_all
                        if (new_diffs - old_diffs) & other_all:
                            continue
                        # Apply
                        rows[ri] = new_row
                        current_max = max_entry(rows)
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        if current_max < best_score:
            best_score = current_max
            best_rows = [list(r) for r in rows]
        return best_rows, best_score

    # Run local search
    best_rows, best_score = try_improve(best_rows, k, time_limit=30.0)

    # If time remains, try multiple greedy starts with different randomizations
    # and pick the best
    if time.time() < deadline - 30.0:
        for seed in range(10):
            if time.time() > deadline - 30.0:
                break
            cand = greedy_rows(n_rows, k, randomize=True, seed=seed)
            cm = max_entry(cand)
            if cm < best_score:
                best_score = cm
                best_rows = cand
                # Try local improvement
                best_rows, best_score = try_improve(best_rows, k, time_limit=10.0)

    # Final check
    if not is_valid(best_rows, k):
        # Fallback to pure greedy
        best_rows = greedy_rows(n_rows, k)

    return best_rows
# EVOLVE-BLOCK-END
