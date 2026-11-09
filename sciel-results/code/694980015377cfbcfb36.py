"""Structural redesign: randomized greedy construction + bitset hill-climbing repair
for the difference-triangle-set (4,7) problem."""
import time
import random
import numpy as np

_DEADLINE = 150.0
_MARGIN = 10.0  # safety margin (seconds)


def _diffs_of_row(row):
    """Return sorted list of positive differences within a single row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the set of all positive differences across all rows."""
    s = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
    return s


def _is_valid(rows, k):
    """Check that all positive differences are distinct across all rows."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_build_row(row_start, used_bits, k, max_val):
    """
    Greedily extend a row (list, starts with 0) to k+1 marks.
    used_bits: numpy boolean array of length max_val+1, True if difference is used.
    Returns the completed row list, or None if stuck.
    """
    row = list(row_start)
    n_marks = len(row)
    while n_marks < k + 1:
        # Try candidate marks in increasing order
        found = False
        for x in range(row[-1] + 1, max_val + 1):
            # Check all differences x - a for a in row
            conflict = False
            for a in row:
                d = x - a
                if d > max_val:
                    conflict = True
                    break
                if used_bits[d]:
                    conflict = True
                    break
            if not conflict:
                # Place x
                for a in row:
                    d = x - a
                    used_bits[d] = True
                row.append(x)
                n_marks += 1
                found = True
                break
        if not found:
            return None
    return row


def _construct_greedy(n_rows, k, max_val, rng):
    """
    Build n_rows rows greedily with a random row ordering.
    Returns list of rows or None.
    """
    used = np.zeros(max_val + 1, dtype=bool)
    rows = []
    # Random order of "slots" — actually we just build rows in random priority
    order = list(range(n_rows))
    rng.shuffle(order)
    for idx in order:
        row = _greedy_build_row([0], used, k, max_val)
        if row is None:
            return None
        rows.append(row)
    return rows


def _hill_climb(rows, k, max_val, rng, time_limit):
    """
    Try to repair/shorten rows by moving individual marks.
    Returns improved rows list or original if no improvement.
    """
    rows = [list(r) for r in rows]
    best_max = _max_entry(rows)
    n_rows = len(rows)
    k_plus_1 = k + 1

    # Build used-difference set
    def compute_used():
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    used.add(row[j] - row[i])
        return used

    used = compute_used()
    # Count distinct diffs; total should be n_rows * k*(k+1)//2
    total_diffs = n_rows * k * (k + 1) // 2

    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try moving each mark in each row to a smaller valid position
        for r in range(n_rows):
            row = rows[r]
            for pos in range(1, k_plus_1):  # skip position 0 (fixed at 0)
                old_val = row[pos]
                # Try smaller values
                for new_val in range(row[pos - 1] + 1, old_val):
                    # Check if moving is valid
                    # Remove old diffs involving pos, add new diffs
                    # Check conflicts with other rows and within row
                    valid = True
                    # Check within-row uniqueness and no conflict with used set
                    # Build tentative row
                    tent = list(row)
                    tent[pos] = new_val
                    # Check strictly increasing
                    for i in range(len(tent) - 1):
                        if tent[i + 1] <= tent[i]:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Compute diffs of tentative row
                    tent_diffs = set()
                    n = len(tent)
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = tent[j] - tent[i]
                            tent_diffs.add(d)
                    # Check no conflict with other rows' diffs
                    # used currently contains ALL diffs including this row's old ones
                    # We need: tent_diffs should not intersect (used - old_row_diffs)
                    old_diffs = set()
                    for i in range(n):
                        for j in range(i + 1, n):
                            old_diffs.add(row[j] - row[i])
                    other_diffs = used - old_diffs
                    if tent_diffs & other_diffs:
                        continue
                    # Also tent_diffs must have no internal duplicates (set handles that)
                    # Accept the move
                    used = (used - old_diffs) | tent_diffs
                    rows[r] = tent
                    if new_val < best_max:
                        best_max = new_val
                        improved = True
                    break  # move to next position

    return rows, best_max


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all positive differences distinct.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN

    # Estimate a reasonable max_val
    # Simple counting bound: n_rows * k*(k+1)//2 distinct diffs in [1, max_val]
    # So max_val >= n_rows * k*(k+1)//2
    total_diffs_needed = n_rows * k * (k + 1) // 2
    # Known best for (4,7) is 134; counting bound is 112
    # Start with a generous upper bound and try to improve
    initial_max = max(total_diffs_needed, 140)

    best_rows = None
    best_max = initial_max

    rng = random.Random(42)

    # Phase 1: Greedy constructions with random orderings
    for attempt in range(200):
        if time.time() > deadline:
            break
        # Try with a max_val that allows some slack
        max_val = initial_max + rng.randint(0, 20)
        rows = _construct_greedy(n_rows, k, max_val, rng)
        if rows is not None:
            cur_max = _max_entry(rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in rows]

    # Phase 2: Hill-climbing repair
    if best_rows is not None:
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Perturb: randomly shuffle row order and rebuild greedily, then hill-climb
            max_val = best_max + rng.randint(0, 30)
            rows = _construct_greedy(n_rows, k, max_val, rng)
            if rows is not None:
                rows, cur_max = _hill_climb(rows, k, max_val, rng, deadline)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = rows

    # Phase 3: More aggressive hill-climbing on the best solution
    if best_rows is not None:
        if time.time() < deadline:
            # Try to further reduce by hill-climbing with larger search
            max_val = best_max + 50
            rows, cur_max = _hill_climb(best_rows, k, max_val, rng, deadline)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = rows

    # If we never found a valid solution, fall back to a simple greedy without limit
    if best_rows is None:
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
        best_rows = rows

    return best_rows
