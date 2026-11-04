"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2

    # Simple counting lower bound: smallest m such that m*(m+1)/2 >= total_diffs
    # Actually the differences are values in 1..m, so we need m*(m+1)/2 >= total_diffs
    lower_bound = 0
    s = 0
    while s < total_diffs:
        lower_bound += 1
        s += lower_bound

    # Try to find a solution with maximum entry as small as possible.
    # Strategy: iterative improvement via randomized greedy + local search.

    best_rows = None
    best_max = float('inf')

    # Helper: greedy construction given a random seed
    def greedy_construct(n_rows, k, rng=None):
        """Greedy: for each row, pick smallest available value that keeps diffs unique."""
        if rng is None:
            rng = random
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

    # Helper: randomized greedy with shuffled candidate ordering
    def randomized_greedy(n_rows, k, rng):
        """
        For each row, instead of always picking the smallest valid x,
        occasionally pick a larger valid x to avoid blocking future rows.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find all valid x values in a window
                valid_candidates = []
                search_limit = x + 50  # look ahead up to 50
                for candidate in range(x + 1, search_limit + 1):
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(candidate)
                        if len(valid_candidates) >= 20:
                            break
                if not valid_candidates:
                    # Fallback: extend search
                    candidate = x + 1
                    while True:
                        new_diffs = {candidate - a for a in row}
                        if not new_diffs & used:
                            valid_candidates.append(candidate)
                            break
                        candidate += 1

                # Pick randomly from valid candidates with bias toward smaller
                if rng.random() < 0.7:
                    # Bias toward smaller values
                    x = valid_candidates[0]
                else:
                    # Pick randomly
                    x = rng.choice(valid_candidates)

                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows

    # Helper: local search / swap improvement
    def local_search(rows, k, max_iters=200):
        """
        Try to reduce the maximum entry by swapping or adjusting values.
        Returns improved rows or None.
        """
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])

        improved = False
        for _ in range(max_iters):
            if _time_left() < 2.0:
                break
            # Find the row with the largest entry
            max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_idx]
            max_val = row[-1]

            # Try to reduce the last element
            found = False
            for new_val in range(row[-2] + 1, max_val):
                # Check if replacing last element with new_val keeps all diffs valid
                # Remove old diffs involving last element
                old_diffs = {row[-1] - a for a in row[:-1]}
                new_diffs = {new_val - a for a in row[:-1]}
                if new_diffs & (used - old_diffs):
                    continue
                # Also check new_diffs don't conflict with each other (they're all positive and distinct since row is increasing)
                # Check that new_val > row[-2]
                if new_val <= row[-2]:
                    continue
                # Valid! Apply
                used -= old_diffs
                used |= new_diffs
                row[-1] = new_val
                found = True
                improved = True
                break

            if not found:
                # Try swapping two elements in the row (not the first)
                # This is more complex; skip for now
                break

        if improved:
            return rows
        return None

    # Run multiple randomized attempts
    rng = random.Random(42)

    # First, try deterministic greedy to get a baseline
    baseline = greedy_construct(n_rows, k)
    best_rows = baseline
    best_max = max(max(r) for r in baseline)

    # Try randomized greedy multiple times
    num_attempts = 50
    for attempt in range(num_attempts):
        if _time_left() < 5.0:
            break
        rows = randomized_greedy(n_rows, k, rng)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
            # Try local search on the best
            if _time_left() > 5.0:
                improved = local_search(best_rows, k, max_iters=100)
                if improved:
                    new_max = max(max(r) for r in improved)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = improved

    # Final local search on best
    if _time_left() > 3.0:
        improved = local_search(best_rows, k, max_iters=500)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    # Validate the result
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    if not validate(best_rows, n_rows, k):
        # Fallback to deterministic greedy
        best_rows = greedy_construct(n_rows, k)

    return best_rows
