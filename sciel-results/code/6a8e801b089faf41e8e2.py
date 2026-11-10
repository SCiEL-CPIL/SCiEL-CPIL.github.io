"""Structural change: backtracking + local search with difference-uniqueness constraints."""
import time
import random
from itertools import combinations


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    # First, get a baseline via greedy
    best_rows = _greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)

    # Try to improve with randomized backtracking and local search
    for attempt in range(200):
        if time.time() > deadline:
            break

        # Randomized backtracking construction
        rows = _randomized_backtrack(n_rows, k, random.Random(attempt))
        if rows is not None:
            current_max = max(max(row) for row in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = rows

        # Local search on the current best
        improved = _local_search(best_rows, n_rows, k, time.time, deadline)
        if improved:
            current_max = max(max(row) for row in improved)
            if current_max < best_max:
                best_max = current_max
                best_rows = improved

    return best_rows


def _greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _randomized_backtrack(n_rows, k, rng):
    """Try to construct rows using backtracking with randomization."""
    used = set()
    rows = [[0] for _ in range(n_rows)]

    # Work on rows one at a time, but with some randomness in choice
    for row_idx in range(n_rows):
        row = rows[row_idx]
        success = _fill_row_backtrack(row, k, used, rng, max_depth=5000)
        if not success:
            return None

    # Verify all differences are unique
    all_diffs = []
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                all_diffs.append(row[j] - row[i])
    if len(all_diffs) != len(set(all_diffs)):
        return None

    return rows


def _fill_row_backtrack(row, k, used, rng, max_depth):
    """Fill a row with k+1 entries using backtracking."""
    if len(row) == k + 1:
        return True

    last = row[-1]
    # Try candidates in random order, biased toward smaller values
    candidates = []
    x = last + 1
    # Look ahead a bit to find valid candidates
    for _ in range(50):
        if x > 200:
            break
        valid = True
        new_diffs = set()
        for a in row:
            d = x - a
            if d in used:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            candidates.append((x, new_diffs))
        x += 1

    if not candidates:
        return False

    # Shuffle with bias toward smaller values
    rng.shuffle(candidates)
    # Sort by value but with some randomness
    candidates.sort(key=lambda c: c[0] + rng.random() * 5)

    for x, new_diffs in candidates[:10]:  # Try top 10 candidates
        row.append(x)
        used.update(new_diffs)
        if _fill_row_backtrack(row, k, used, rng, max_depth):
            return True
        row.pop()
        used.difference_update(new_diffs)

    return False


def _local_search(rows, n_rows, k, time_func, deadline):
    """Local search: try to reduce the maximum value by modifying entries."""
    best_rows = [list(row) for row in rows]
    best_max = max(max(row) for row in best_rows)
    rng = random.Random()

    for iteration in range(10000):
        if time_func() > deadline:
            break

        # Pick a random row and a random position to modify
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)  # Don't modify the 0

        original = best_rows[row_idx][pos]

        # Try to replace with a smaller value
        # Find the smallest valid value that's smaller than original
        row = best_rows[row_idx]
        lower_bound = row[pos - 1] + 1 if pos > 0 else 1
        upper_bound = row[pos + 1] - 1 if pos < k else original - 1

        if upper_bound < lower_bound:
            continue

        # Try values in random order, biased toward smaller
        candidates = list(range(lower_bound, min(upper_bound, original) + 1))
        rng.shuffle(candidates)

        for new_val in candidates[:20]:
            # Check if this value is valid
            if _is_valid_for_position(best_rows, row_idx, pos, new_val):
                # Accept this move
                best_rows[row_idx][pos] = new_val
                current_max = max(max(row) for row in best_rows)
                if current_max < best_max:
                    best_max = current_max
                break
        else:
            # No improvement found, revert
            best_rows[row_idx][pos] = original

    return best_rows


def _is_valid_for_position(rows, row_idx, pos, new_val):
    """Check if placing new_val at position pos in row_idx is valid."""
    row = rows[row_idx]

    # Check increasing property
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < len(row) - 1 and new_val >= row[pos + 1]:
        return False

    # Check difference uniqueness
    # Remove the old value's differences
    old_val = row[pos]
    old_diffs = set()
    for i in range(len(row)):
        if i != pos:
            d = abs(row[i] - old_val)
            old_diffs.add(d)

    # Add the new value's differences
    new_diffs = set()
    for i in range(len(row)):
        if i != pos:
            d = abs(row[i] - new_val)
            new_diffs.add(d)

    # Check if new_diffs conflicts with any differences from other rows or other positions
    # Build the set of all differences except those involving (row_idx, pos)
    all_diffs = set()
    for r_idx in range(len(rows)):
        r = rows[r_idx]
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                if r_idx == row_idx and (i == pos or j == pos):
                    continue
                all_diffs.add(r[j] - r[i])

    # Check if new_diffs intersects with all_diffs
    if new_diffs & all_diffs:
        return False

    return True


def _validate(rows, n_rows, k):
    """Validate that the rows satisfy all constraints."""
    # Check each row has k+1 entries, starts at 0, strictly increasing
    for i, row in enumerate(rows):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for j in range(1, len(row)):
            if row[j] <= row[j - 1]:
                return False

    # Check all positive differences are distinct
    all_diffs = []
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                all_diffs.append(row[j] - row[i])

    return len(all_diffs) == len(set(all_diffs))


# Ensure the returned rows are valid
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    # First, get a baseline via greedy
    best_rows = _greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)

    # Verify baseline
    if not _validate(best_rows, n_rows, k):
        # Fallback: simple row-by-row greedy
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                diffs = {x - a for a in row}
                if not (diffs & used):
                    row.append(x)
                    used |= diffs
                x += 1
            best_rows.append(row)
        best_max = max(max(row) for row in best_rows)

    # Try to improve with randomized backtracking and local search
    for attempt in range(300):
        if time.time() > deadline:
            break

        # Randomized backtracking construction
        rows = _randomized_backtrack(n_rows, k, random.Random(attempt * 7 + 13))
        if rows is not None and _validate(rows, n_rows, k):
            current_max = max(max(row) for row in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(row) for row in rows]

        # Local search on the current best
        if time.time() < deadline - 5.0:
            improved = _local_search(best_rows, n_rows, k, time.time, deadline)
            if improved and _validate(improved, n_rows, k):
                current_max = max(max(row) for row in improved)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(row) for row in improved]

    # Final validation
    if not _validate(best_rows, n_rows, k):
        # Absolute fallback
        best_rows = _greedy_construct(n_rows, k)

    return best_rows
