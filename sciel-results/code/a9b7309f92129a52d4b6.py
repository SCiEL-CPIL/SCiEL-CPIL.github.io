"""
Difference triangle set (n_rows, k) constructor.

Strategy:
- Fast construction via a difference-aware greedy (with small randomized perturbations)
  to obtain valid starting rows quickly.
- Iterative local-search "tightening" with efficient conflict accounting:
  * Maintain per-difference counts and the list of rows/diffs that are currently conflicting.
  * For each candidate move (change one element in one row), compute the delta in conflicts
    and the new max entry, then decide acceptance via a simple SA-like rule that strongly
    penalizes conflicts but allows occasional uphill moves to escape local minima.
- Explicit deadline guard to return best valid solution found before 150 s.
"""
import random
import time
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Hard deadline: return before 150 s wall-clock with a safety margin.
    deadline = start_time + 145.0

    # Each row has k+1 entries (including 0). Differences per row = k*(k+1)/2.
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on the maximum entry: at least total_diffs (simple counting bound)
    lower_bound = total_diffs

    # Helper: compute all positive differences in a row
    def row_diffs(row):
        diffs = []
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - ri)
        return diffs

    # Helper: validate rows (all diffs distinct)
    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    # Helper: get max entry
    def get_max_entry(rows):
        m = 0
        for row in rows:
            if row and row[-1] > m:
                m = row[-1]
        return m

    # Greedy construction with a small lookahead / randomization to diversify starts
    def greedy_construct(max_val_hint=None, seed=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            # Choose k increasing integers from 1..max_val_hint
            # Use a greedy-with-lookahead: try smallest available that doesn't create conflicts
            # But to diversify, occasionally skip the smallest and pick a larger one
            for pos in range(k):
                # Find the smallest x > row[-1] such that all (x - a) for a in row are unused
                # And also (x - a) doesn't conflict with used
                found = False
                # Try a window of candidates
                start = row[-1] + 1
                if max_val_hint is not None:
                    end = max_val_hint + 1
                else:
                    end = start + 200  # small window
                # Collect candidate values in order
                candidates = []
                for x in range(start, end):
                    ok = True
                    for a in row:
                        d = x - a
                        if d in used:
                            ok = False
                            break
                    if ok:
                        candidates.append(x)
                        if len(candidates) >= 5:
                            break
                if candidates:
                    # Pick the first (greedy) or a random one for diversity
                    if rng.random() < 0.2 and len(candidates) > 1:
                        x = rng.choice(candidates)
                    else:
                        x = candidates[0]
                    row.append(x)
                    for a in row:
                        used.add(x - a)
                    found = True
                else:
                    # Fallback: search wider
                    for x in range(start, start + 5000):
                        ok = True
                        for a in row:
                            d = x - a
                            if d in used:
                                ok = False
                                break
                        if ok:
                            row.append(x)
                            for a in row:
                                used.add(x - a)
                            found = True
                            break
                if not found:
                    # Shouldn't happen for reasonable parameters
                    row.append(row[-1] + 1)
            rows.append(row)
        return rows

    # Initialize with greedy
    best_rows = greedy_construct(seed=42)
    if not is_valid(best_rows):
        # Fallback: simple sequential greedy
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for pos in range(k):
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                row.append(x)
                for a in row:
                    used.add(x - a)
            rows.append(row)
        best_rows = rows

    best_max = get_max_entry(best_rows)

    # Local search / SA tightening
    # We maintain:
    #   rows: current state
    #   diff_count: dict diff -> count (how many rows have this diff)
    #   We want diff_count[d] == 1 for all d used, and 0 for others.
    #   conflicts = sum(max(0, diff_count[d] - 1) for d in diff_count)

    def build_diff_counts(rows):
        counts = {}
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def count_conflicts(counts):
        conf = 0
        for c in counts.values():
            if c > 1:
                conf += c - 1
        return conf

    # SA parameters
    T_start = 50.0
    T_end = 0.5
    cooling = 0.9995
    max_iterations_per_restart = 200000

    # Multiple restarts with different initializations
    num_restarts = 20

    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        # New random start
        if restart == 0:
            current_rows = [row[:] for row in best_rows]
        else:
            current_rows = greedy_construct(seed=1000 + restart * 7)
            if not is_valid(current_rows):
                # Try to fix by simple greedy rebuild
                used = set()
                rows = []
                for r in range(n_rows):
                    row = [0]
                    for pos in range(k):
                        x = row[-1] + 1
                        while True:
                            ok = True
                            for a in row:
                                if (x - a) in used:
                                    ok = False
                                    break
                            if ok:
                                break
                            x += 1
                        row.append(x)
                        for a in row:
                            used.add(x - a)
                    rows.append(row)
                current_rows = rows

        curr_max = get_max_entry(current_rows)
        if curr_max < best_max and is_valid(current_rows):
            best_max = curr_max
            best_rows = [row[:] for row in current_rows]

        # Build diff counts
        diff_counts = build_diff_counts(current_rows)
        curr_conflicts = count_conflicts(diff_counts)

        # If already valid and better, we can skip SA for this restart
        if curr_conflicts == 0 and curr_max < best_max:
            continue

        T = T_start
        iters = 0
        stagnant_count = 0
        max_stagnant = 5000

        while T > T_end and time.time() < deadline - 2.0 and iters < max_iterations_per_restart:
            iters += 1

            # Pick a random row and a random element index (1..k, not 0)
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)

            row = current_rows[r_idx]
            old_val = row[c_idx]

            # Determine range for new value
            min_val = row[c_idx - 1] + 1
            max_val = curr_max + 20  # allow some room to explore
            if c_idx < k:
                max_val = min(max_val, row[c_idx + 1] - 1)

            if min_val > max_val:
                T *= cooling
                continue

            # Choose new value: bias towards smaller values to tighten
            # 70% chance: pick from [min_val, min(old_val, max_val)]
            # 30% chance: pick from [min_val, max_val]
            if random.random() < 0.7:
                upper = min(old_val, max_val)
                if min_val > upper:
                    new_val = old_val
                else:
                    new_val = random.randint(min_val, upper)
            else:
                new_val = random.randint(min_val, max_val)

            if new_val == old_val:
                T *= cooling
                continue

            # Compute delta in conflicts
            # Remove old differences involving position c_idx
            old_diffs = []
            for i in range(len(row)):
                if i == c_idx:
                    continue
                d = abs(row[c_idx] - row[i])
                # Actually differences are row[j]-row[i] for i<j, so:
                if i < c_idx:
                    d = row[c_idx] - row[i]
                else:
                    d = row[i] - row[c_idx]
                old_diffs.append(d)

            # Remove old
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] <= 0:
                    del diff_counts[d]

            # Apply new value
            row[c_idx] = new_val

            # Add new differences
            new_diffs = []
            for i in range(len(row)):
                if i == c_idx:
                    continue
                if i < c_idx:
                    d = row[c_idx] - row[i]
                else:
                    d = row[i] - row[c_idx]
                new_diffs.append(d)

            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1

            new_conflicts = count_conflicts(diff_counts)
            new_max = max(curr_max, row[c_idx])
            # Recompute max properly if we reduced the global max
            if new_max == curr_max and row[c_idx] < old_val:
                # Check if old_val was the unique max
                # For simplicity, recompute if needed
                new_max = get_max_entry(current_rows)

            # Cost function: conflicts * penalty + max_entry
            # Penalty should be large enough to prefer valid solutions
            conflict_penalty = 1000
            old_cost = curr_conflicts * conflict_penalty + curr_max
            new_cost = new_conflicts * conflict_penalty + new_max

            delta_cost = new_cost - old_cost

            # Acceptance
            if delta_cost < 0 or (T > 0.01 and random.random() < math.exp(-delta_cost / T)):
                # Accept
                curr_max = new_max
                curr_conflicts = new_conflicts
                if new_conflicts == 0 and new_max < best_max:
                    best_max = new_max
                    best_rows = [row[:] for row in current_rows]
                    stagnant_count = 0
                else:
                    if new_conflicts == 0 and new_max == best_max:
                        stagnant_count += 1
                    else:
                        stagnant_count += 1
            else:
                # Reject: revert
                row[c_idx] = old_val
                # Revert diff counts
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] <= 0:
                        del diff_counts[d]
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1

            T *= cooling

            # Early stop if stagnant
            if stagnant_count > max_stagnant:
                break

        # End of SA for this restart

    # Final validation
    if not is_valid(best_rows):
        # Fallback to a known valid solution
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for pos in range(k):
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                row.append(x)
                for a in row:
                    used.add(x - a)
            rows.append(row)
        best_rows = rows

    return best_rows
