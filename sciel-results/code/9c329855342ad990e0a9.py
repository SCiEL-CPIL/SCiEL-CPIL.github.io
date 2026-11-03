"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows difference triangle sets with k+1 entries per row.
    Each row starts at 0 and strictly increases. All positive differences
    across all rows must be distinct. Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Total number of differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope (largest entry)
    # We need total_diffs distinct positive integers, each <= scope
    # The smallest possible set of total_diffs distinct positive ints is 1..total_diffs
    # So scope >= total_diffs, but also each row's max is at least the sum of its gaps
    # A tighter bound: for one row with k+1 points, max >= k*(k+1)/2 (if all gaps are 1)
    # But since differences must be unique across rows, we need more.
    # Simple counting bound: scope >= total_diffs (since we need total_diffs distinct values in [1, scope])
    # Actually, the differences are all in [1, scope], and there are total_diffs of them, all distinct.
    # So scope >= total_diffs.
    # But also, for a single row, the largest difference is the row's span (last - first = last since first=0).
    # The row has diffs_per_row differences, all <= span. So span >= diffs_per_row.
    # And span <= scope. So scope >= diffs_per_row.
    # The counting bound scope >= total_diffs is usually tighter.

    min_scope = total_diffs  # counting bound

    # Try scopes from min_scope upward
    # Use backtracking with bitset for used differences

    # For efficiency, we'll try to construct rows one by one.
    # For each row, we need to choose k+1 values starting at 0.
    # The differences within a row must not conflict with used differences.

    # Strategy: Try to build each row greedily with backtracking if needed.
    # Start with a greedy approach, and if it fails for a given scope, try next scope.

    best_result = None
    best_scope = float('inf')

    # Try increasing scopes
    for scope in range(min_scope, min_scope + 200):
        if time.time() > deadline:
            break

        # Try to construct n_rows rows with max entry <= scope
        result = _construct_with_scope(n_rows, k, scope, time.time(), deadline)
        if result is not None:
            best_result = result
            best_scope = scope
            break  # Found a valid construction with this scope, which is minimal in our search

    if best_result is None:
        # Fallback: greedy without scope constraint
        best_result = _greedy_construct(n_rows, k)

    return best_result


def _construct_with_scope(n_rows, k, scope, start_time, deadline):
    """
    Try to construct n_rows rows with k+1 entries each, starting at 0,
    strictly increasing, max entry <= scope, and all differences distinct.
    """
    diffs_per_row = k * (k + 1) // 2

    # Bitset of used differences. Differences range from 1 to scope.
    # Use a Python set for simplicity, or a bitset via integer.
    # Since scope can be up to ~2000, a set is fine.
    used = set()
    rows = []

    # We'll build rows one by one. For each row, we choose k values (the non-zero entries).
    # The row is [0, x1, x2, ..., xk] with 0 < x1 < x2 < ... < xk <= scope.
    # The differences are all xj - xi for i < j, including xj - 0 = xj.

    # To make this tractable, we can try a greedy approach with backtracking.
    # For each row, try to find a valid set of k values.

    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None

        # Try to construct this row
        row = _construct_row(k, scope, used, time.time(), deadline)
        if row is None:
            return None

        rows.append(row)
        # Update used differences
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])

    return rows


def _construct_row(k, scope, used, start_time, deadline):
    """
    Construct a single row of k+1 entries starting at 0, strictly increasing,
    max <= scope, such that all differences are not in 'used'.
    """
    # Backtracking search for the k non-zero entries
    # We choose x1 < x2 < ... < xk, all in [1, scope]
    # Differences: for each pair (i, j) with i < j in {0, x1, ..., xk}, the diff must not be in used

    # To speed up, we can precompute which values are "available" (not in used)
    # But the differences depend on the chosen values, so we need to check dynamically.

    # Use backtracking with forward checking
    def backtrack(pos, current_row, used_so_far):
        if time.time() > deadline:
            return None

        if pos == k:
            # Check that all differences are valid
            row = [0] + current_row
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in diffs:
                        return None
                    diffs.add(d)
            return row

        # Try to choose the next value
        # The next value must be greater than the last chosen (or 1 if pos==0)
        # and <= scope
        last = current_row[-1] if current_row else 0
        start_val = last + 1

        for val in range(start_val, scope + 1):
            if time.time() > deadline:
                return None

            # Check if val itself is a valid difference (val - 0 = val)
            if val in used:
                continue

            # Check differences with all previously chosen values
            valid = True
            for prev in current_row:
                d = val - prev
                if d in used:
                    valid = False
                    break
            if not valid:
                continue

            # Tentatively add val and check if we can complete
            current_row.append(val)
            # Update used_so_far with new differences
            new_diffs = set()
            for prev in current_row[:-1]:  # all except the last one just added
                new_diffs.add(val - prev)
            new_diffs.add(val)  # val - 0

            if new_diffs & used:
                current_row.pop()
                continue

            used_so_far.update(new_diffs)
            result = backtrack(pos + 1, current_row, used_so_far)
            if result is not None:
                return result
            used_so_far.difference_update(new_diffs)
            current_row.pop()

        return None

    return backtrack(0, [], set())


def _greedy_construct(n_rows, k):
    """
    Greedy fallback: each row starts at 0 and takes the smallest next mark
    whose differences to the earlier marks of the row are all still unused by any row.
    """
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
# EVOLVE-BLOCK-END
