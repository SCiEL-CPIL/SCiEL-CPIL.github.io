import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # The task is for (n_rows, k) = (7, 5) based on the description.
    # Each row has k+1 = 6 entries, starting at 0.
    # We need to find 7 rows of 6 increasing integers starting at 0
    # such that all pairwise differences within each row are distinct across all rows.
    # Minimize the largest entry.

    # For (7, 5): each row has 6 elements, so C(6,2) = 15 differences per row.
    # Total 7*15 = 105 distinct differences needed.
    # The counting bound is 105, meaning if we use differences 1..105 exactly,
    # the max entry would be at least 105 (since 0 is included and max diff = max entry).
    # Best known is 112.

    # Strategy: Use a backtracking search with heuristics.
    # We'll try to build rows one at a time, choosing entries greedily with backtracking.

    # First, let's try a smarter greedy approach with limited backtracking.
    # Then if time permits, do local search to improve.

    def get_diffs(row):
        """Get all pairwise differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_row(row, used_diffs):
        """Check if a row's differences don't conflict with used_diffs."""
        return not (get_diffs(row) & used_diffs)

    def greedy_row(used_diffs, k, max_val):
        """Greedily build a row of k+1 elements starting at 0."""
        row = [0]
        while len(row) < k + 1:
            found = False
            for x in range(row[-1] + 1, max_val + 1):
                candidate = row + [x]
                if is_valid_row(candidate, used_diffs):
                    row = candidate
                    found = True
                    break
            if not found:
                return None
        return row

    # Try greedy first to get a baseline
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used_diffs, k, 200)
        if row is None:
            break
        rows.append(row)
        used_diffs |= get_diffs(row)

    if len(rows) == n_rows:
        current_max = max(max(r) for r in rows)
    else:
        current_max = float('inf')

    # Now try to improve with backtracking search
    # We'll use a recursive backtracking that tries to place rows optimally

    def backtrack(rows_partial, used_diffs, row_idx, max_val_bound):
        """Try to complete the set of rows starting from row_idx."""
        if time.time() > deadline:
            return None, float('inf')

        if row_idx == n_rows:
            return list(rows_partial), max(max(r) for r in rows_partial)

        best_result = None
        best_max = float('inf')

        # Try to build a row starting at 0 with k+1 elements
        # Use backtracking within the row construction too
        def build_row(current_row, pos, used_diffs, max_val_bound):
            """Recursively build a row, returning (row, new_used_diffs) or None."""
            if time.time() > deadline:
                return None, None

            if len(current_row) == k + 1:
                return list(current_row), get_diffs(current_row)

            # Try next value
            start_val = current_row[-1] + 1 if current_row else 1
            for x in range(start_val, max_val_bound + 1):
                candidate = current_row + [x]
                cand_diffs = get_diffs(candidate)
                if not (cand_diffs & used_diffs):
                    result, new_diffs = build_row(candidate, pos + 1, used_diffs, max_val_bound)
                    if result is not None:
                        return result, new_diffs
            return None, None

        # Try different max values, starting from a reasonable lower bound
        # The minimum possible max for this row given used_diffs
        min_possible = max(used_diffs) if used_diffs else 1

        for max_val in range(min_possible, min_possible + 50):  # Try a range
            if time.time() > deadline:
                break
            row, new_diffs = build_row([0], 0, used_diffs, max_val)
            if row is not None:
                result, result_max = backtrack(rows_partial + [row], used_diffs | new_diffs, row_idx + 1, max_val)
                if result is not None and result_max < best_max:
                    best_max = result_max
                    best_result = result
                    # If we hit the counting bound, we can stop
                    if best_max <= 105:
                        return best_result, best_max

        return best_result, best_max

    # Try backtracking from scratch
    result, result_max = backtrack([], set(), 0, 105)
    if result is not None and result_max < current_max:
        rows = result
        current_max = result_max

    # If we have a valid solution, return it
    # Verify validity
    all_diffs = set()
    valid = True
    for row in rows:
        if len(row) != k + 1:
            valid = False
            break
        if row[0] != 0:
            valid = False
            break
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    valid = False
                    break
                all_diffs.add(d)
        if not valid:
            break

    if not valid or len(rows) != n_rows:
        # Fallback: return the greedy result if it's valid
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used_diffs, k, 300)
            if row is None:
                # Return what we have padded with large values
                row = [0] + list(range(1, k+1))
                # This won't be valid but we need to return something
                rows.append(row)
                continue
            rows.append(row)
            used_diffs |= get_diffs(row)

    return rows
