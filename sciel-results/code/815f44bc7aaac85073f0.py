import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    best_max = float('inf')
    best_rows = None

    # --- Helper Functions ---

    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_max(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    # --- Greedy Baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = compute_max(best_rows)

    # --- Difference-space backtracking solver ---
    # Each row is represented by its k+1 entries [0, a1, a2, ..., ak].
    # The differences are a1, a2-a1, ..., ak-a_{k-1} (consecutive gaps) plus
    # all pairwise differences. We search over gap sequences that are
    # strictly positive and whose cumulative sums give a valid row.
    # Constraint: all pairwise differences across all rows must be distinct.

    def row_from_gaps(gaps):
        """Given k positive gaps, return the row of k+1 entries starting at 0."""
        row = [0]
        s = 0
        for g in gaps:
            s += g
            row.append(s)
        return row

    def row_diffs_from_gaps(gaps):
        """Return the set of all pairwise differences from a row defined by gaps."""
        row = row_from_gaps(gaps)
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def backtrack_csp(used_diffs, n_remaining, max_val, depth=0):
        """
        Recursive backtracking: assign gap sequences to rows.
        used_diffs: set of differences already used.
        n_remaining: number of rows left to assign.
        max_val: current maximum entry (for scope tracking).
        Returns (rows, max_val) or (None, None) on failure.
        """
        if time.time() > deadline - 3.0:
            return None, None

        if n_remaining == 0:
            return [], max_val

        # Try different gap sequences, ordered by increasing max value
        # Generate candidate gaps: all compositions of some total into k positive parts
        # We'll try small totals first to minimize scope.
        # For efficiency, we iterate over possible max values.

        # Start from max_val+1 and go up, but cap the search
        # Use a heuristic: try gaps that are "spread out" to minimize collisions
        found = False
        rows_found = []
        new_max = max_val

        # Try different total sums starting from the minimum possible
        min_total = k * (k + 1) // 2  # minimum sum of k positive integers that give k+1 entries
        # Actually minimum total for k gaps giving distinct diffs is k*(k+1)/2
        # But we need to be more careful. Let's just try increasing totals.

        # For each possible total, try random permutations of gap sequences
        # To keep this tractable, we'll use a limited search.
        max_total_to_try = max_val + 40 if max_val > 0 else 30

        for total in range(max(min_total, max_val + 1), max_total_to_try + 1):
            if time.time() > deadline - 3.0:
                break

            # Generate a random gap sequence summing to total
            # Use stars and bars with random splits
            for attempt in range(20):
                if time.time() > deadline - 3.0:
                    break

                # Random composition of total into k positive parts
                if k == 1:
                    gaps = [total]
                else:
                    # Random split
                    splits = sorted(random.sample(range(1, total), k - 1))
                    gaps = []
                    prev = 0
                    for s in splits:
                        gaps.append(s - prev)
                        prev = s
                    gaps.append(total - prev)

                # Check all gaps are positive (guaranteed by construction)
                # Compute row and its diffs
                row_diffs = row_diffs_from_gaps(gaps)

                # Check no collision with used_diffs
                if row_diffs & used_diffs:
                    continue

                # Valid! Recurse
                new_used = used_diffs | row_diffs
                row = row_from_gaps(gaps)
                new_max_val = max(new_max, row[-1])

                sub_rows, sub_max = backtrack_csp(new_used, n_remaining - 1, new_max_val, depth + 1)
                if sub_rows is not None:
                    rows_found = [row] + sub_rows
                    found = True
                    break

            if found:
                return rows_found, max(new_max, rows_found[0][-1])

        return None, None

    def csp_solve():
        """Run the CSP backtracking solver."""
        start_time = time.time()
        used = set()
        rows, max_val = backtrack_csp(used, n_rows, 0)
        if rows is not None:
            return rows, max_val
        return None, None

    # --- Polish Phase: try to reduce the maximum entry ---
    def polish(rows, time_limit):
        rows = [r[:] for r in rows]
        current_max = compute_max(rows)
        start_time = time.time()

        diff_counts = defaultdict(int)
        for r in rows:
            for d in get_diffs(r):
                diff_counts[d] += 1

        # Check if valid
        for d, c in diff_counts.items():
            if c > 1:
                return rows, current_max

        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try to reduce the last element of each row
            for r_idx in range(n_rows):
                row = rows[r_idx]
                old_last = row[-1]
                # Try values from old_last-1 down to row[-2]+1
                for new_val in range(old_last - 1, row[-2], -1):
                    old_diffs_to_remove = [old_last - x for x in row[:-1]]
                    new_diffs_to_add = [new_val - x for x in row[:-1]]

                    # Check if new diffs are all unused
                    valid = True
                    for d in new_diffs_to_add:
                        if d in diff_counts:
                            valid = False
                            break

                    if valid:
                        # Remove old diffs
                        for d in old_diffs_to_remove:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        # Add new diffs
                        for d in new_diffs_to_add:
                            diff_counts[d] = 1
                        row[-1] = new_val
                        improved = True
                        current_max = compute_max(rows)
                        break

        return rows, current_max

    # --- Simulated Annealing for refinement ---
    def sa_optimize(initial_rows, time_limit, seed_offset=0):
        random.seed(seed_offset)
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len

        # Precompute diffs per row and global counts
        row_diffs = [get_diffs(r) for r in rows]
        diff_counts = defaultdict(int)
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] += 1

        # Calculate initial collision count
        collision_count = 0
        for d, c in diff_counts.items():
            if c > 1:
                collision_count += c - 1

        current_max = compute_max(rows)
        W_COLL = 10000

        def get_cost():
            return collision_count * W_COLL + current_max

        current_cost = get_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max

        T = 200.0
        cooling = 0.99995
        min_T = 0.5
        start_time = time.time()

        while time.time() - start_time < time_limit:
            # Select random row and index
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)

            row = rows[r_idx]
            old_val = row[i_idx]

            # Determine valid range for new_val
            lower = row[i_idx - 1] + 1
            if i_idx < L - 1:
                upper = row[i_idx + 1] - 1
            else:
                upper = current_max + 30

            if lower > upper:
                continue

            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue

            # --- Update State ---
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                c = diff_counts[d]
                if c == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = c - 1
                    if c >= 2:
                        collision_count -= 1

            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs

            for d in new_diffs:
                c = diff_counts.get(d, 0)
                if c == 0:
                    diff_counts[d] = 1
                else:
                    diff_counts[d] = c + 1
                    if c >= 1:
                        collision_count += 1

            new_global_max = compute_max(rows)
            new_cost = collision_count * W_COLL + new_global_max

            delta = new_cost - current_cost
            accept = False

            if delta < 0:
                accept = True
            else:
                if T > min_T:
                    p = math.exp(-delta / T)
                    if random.random() < p:
                        accept = True
                elif delta == 0:
                    if random.random() < 0.3:
                        accept = True

            if accept:
                current_cost = new_cost
                current_max = new_global_max
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = new_global_max
            else:
                row[i_idx] = old_val
                row_diffs[r_idx] = old_diffs

                for d in old_diffs:
                    c = diff_counts.get(d, 0)
                    if c == 0:
                        diff_counts[d] = 1
                    else:
                        diff_counts[d] = c + 1
                        if c >= 1:
                            collision_count += 1

                for d in new_diffs:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                        if c >= 2:
                            collision_count -= 1

                current_max = compute_max(rows)

            T *= cooling
            if T < min_T:
                T = min_T

        return best_local_rows, best_local_max

    # --- Main Loop ---
    remaining = deadline - time.time()

    # First, try CSP solver with a time budget
    csp_budget = min(remaining * 0.4, 40.0)
    if csp_budget > 5.0:
        csp_start = time.time()
        csp_rows, csp_max = csp_solve()
        if csp_rows is not None and is_valid(csp_rows):
            if csp_max < best_max:
                best_rows = csp_rows
                best_max = csp_max

    # Then use SA to refine
    sa_remaining = deadline - time.time() - 5.0
    if sa_remaining > 5.0:
        n_starts = 4
        time_per_start = sa_remaining / n_starts

        for s in range(n_starts):
            if time.time() >= deadline - 5.0:
                break

            if s == 0:
                start_rows = best_rows
            else:
                start_rows = [r[:] for r in best_rows]
                for _ in range(3):
                    r_idx = random.randint(0, n_rows - 1)
                    i_idx = random.randint(1, row_len - 1)
                    old_val = start_rows[r_idx][i_idx]
                    lower = start_rows[r_idx][i_idx - 1] + 1
                    if i_idx < row_len - 1:
                        upper = start_rows[r_idx][i_idx + 1] - 1
                    else:
                        upper = compute_max(start_rows) + 10
                    if lower <= upper:
                        start_rows[r_idx][i_idx] = random.randint(lower, upper)

            t_remaining = deadline - time.time() - 5.0
            if t_remaining <= 0:
                break

            local_rows, local_max = sa_optimize(start_rows, t_remaining, seed_offset=42 + s * 1000)

            if is_valid(local_rows):
                if local_max < best_max:
                    best_rows = local_rows
                    best_max = local_max

    # Final polish
    if time.time() < deadline - 1.0:
        polish_time = deadline - time.time() - 1.0
        polished_rows, polished_max = polish(best_rows, polish_time)
        if polished_max < best_max:
            best_rows = polished_rows
            best_max = polished_max

    return best_rows
