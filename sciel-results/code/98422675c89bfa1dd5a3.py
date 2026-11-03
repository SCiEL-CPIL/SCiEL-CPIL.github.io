import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Difference triangle set construction.
    Each row has k+1 entries starting at 0, strictly increasing.
    All positive differences a_j - a_i (i < j) across all rows must be distinct.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Total number of differences per row: C(k+1, 2)
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Counting bound: we need total_diffs distinct positive integers.
    # The smallest possible max difference is at least total_diffs.
    # But the max entry M must satisfy that all differences are <= M.
    # The number of distinct differences <= M is M (values 1..M).
    # So we need M >= total_diffs. But also structural constraints.
    # Simple counting bound: M >= total_diffs, but actually we need M such that
    # we can place all diffs in [1, M]. The counting bound is M >= total_diffs.
    # For (7,5): diffs_per_row = 15, total_diffs = 105. Best known is 112.

    # Strategy: Use simulated annealing / local search.
    # Represent each row as a sorted list of k+1 integers starting with 0.
    # Objective: minimize max entry, subject to all differences being distinct.
    # 
    # Approach:
    # 1. Start with a greedy construction to get a valid initial solution.
    # 2. Use local search: try to reduce the maximum entry by moving elements.
    # 3. Use simulated annealing to escape local optima.

    def row_diffs(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            for d in row_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    def max_entry(rows):
        return max(max(row) for row in rows)

    # Step 1: Greedy initial construction
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    rows = greedy_construct(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_score = max_entry(best_rows)

    if time.time() > deadline:
        return best_rows

    # Step 2: Local search with simulated annealing
    # We want to minimize the maximum entry.
    # Move: pick a row, pick a position (not 0), try to decrease or increase the value.
    # We need to maintain: strictly increasing, starts at 0, all diffs distinct.

    def try_move(rows, row_idx, pos, new_val, used_diffs):
        """
        Try to set rows[row_idx][pos] = new_val.
        Returns (success, new_used_diffs, new_max) or None.
        used_diffs is a set of all differences currently in use.
        """
        row = rows[row_idx]
        old_val = row[pos]
        if new_val == old_val:
            return None
        # Check strictly increasing
        if pos > 0 and new_val <= row[pos - 1]:
            return None
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return None

        # Compute old diffs for this row
        old_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                old_diffs.add(row[j] - row[i])

        # Remove old diffs from used_diffs
        new_used = used_diffs - old_diffs

        # Compute new diffs for this row
        row[pos] = new_val
        new_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                new_diffs.add(d)
                if d in new_used:
                    row[pos] = old_val
                    return None
                new_used.add(d)
        row[pos] = old_val

        new_max = max(max(r) for r in rows)
        return (True, new_used, new_max)

    # Simulated annealing
    T = 100.0  # initial temperature
    cooling = 0.999
    min_T = 0.01

    # Precompute used_diffs
    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            used |= row_diffs(row)
        return used

    used_diffs = compute_used_diffs(rows)
    current_max = max_entry(rows)

    random.seed(42)

    while time.time() < deadline - 5:
        # Pick a random row and position
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)  # positions 1..k (0 is fixed)

        # Try to decrease the value (to reduce max)
        row = rows[row_idx]
        old_val = row[pos]

        # Try decreasing by a random amount
        max_decrease = old_val - (row[pos - 1] + 1)
        if max_decrease > 0:
            dec = random.randint(1, min(max_decrease, 10))
            new_val = old_val - dec
        else:
            # Try increasing slightly (might help rearrange)
            new_val = old_val + random.randint(1, 5)

        result = try_move(rows, row_idx, pos, new_val, used_diffs)
        if result is None:
            continue

        success, new_used, new_max = result

        if new_max < current_max:
            # Accept and update
            rows[row_idx][pos] = new_val
            used_diffs = new_used
            current_max = new_max
            if current_max < best_score:
                best_score = current_max
                best_rows = [list(r) for r in rows]
        else:
            # Simulated annealing: accept with probability exp(-delta/T)
            delta = new_max - current_max
            if delta > 0:
                prob = math.exp(-delta / max(T, 0.001))
                if random.random() < prob:
                    rows[row_idx][pos] = new_val
                    used_diffs = new_used
                    current_max = new_max
            # else reject

        T *= cooling

        if T < min_T:
            T = min_T

    return best_rows
# EVOLVE-BLOCK-END
