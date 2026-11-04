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

    # Helper: improved greedy that considers future rows
    def lookahead_greedy(n_rows, k, rng):
        """
        Greedy with lookahead: when choosing a value, consider how many
        future valid options it leaves for subsequent positions.
        """
        used = set()
        rows = []
        for row_idx in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find several valid candidates
                valid_candidates = []
                search_limit = x + 60
                for candidate in range(x + 1, search_limit + 1):
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(candidate)
                        if len(valid_candidates) >= 10:
                            break
                if not valid_candidates:
                    candidate = x + 1
                    while True:
                        new_diffs = {candidate - a for a in row}
                        if not new_diffs & used:
                            valid_candidates.append(candidate)
                            break
                        candidate += 1

                # Score each candidate by how many options it leaves for next step
                best_score = -1
                best_cand = valid_candidates[0]
                for cand in valid_candidates[:8]:  # Only score top 8
                    new_diffs = {cand - a for a in row}
                    # Count how many valid next values would exist
                    next_count = 0
                    for next_cand in range(cand + 1, cand + 30):
                        next_diffs = {next_cand - a for a in row}
                        next_diffs.add(next_cand - cand)
                        if not (next_diffs & (used | new_diffs)):
                            next_count += 1
                        if next_count >= 5:
                            break
                    # Prefer smaller values with good future options
                    score = next_count * 100 - cand
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                x = best_cand
                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows

    # Helper: randomized greedy with shuffled candidate ordering
    def randomized_greedy(n_rows, k, rng):
        """
        For each row, instead of always picking the smallest valid x,
        occasionally pick a larger valid x to avoid blocking future rows.
        Uses adaptive search window based on progress.
        """
        used = set()
        rows = []
        for row_idx in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find all valid x values in a window
                # Use a wider window for earlier rows, narrower for later
                window = max(30, 80 - 10 * row_idx)
                valid_candidates = []
                search_limit = x + window
                for candidate in range(x + 1, search_limit + 1):
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(candidate)
                        if len(valid_candidates) >= 15:
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
                # Use a softmax-like selection: probability proportional to 1/(rank+1)
                n_cand = len(valid_candidates)
                if n_cand == 1:
                    x = valid_candidates[0]
                else:
                    # Weighted random: favor smaller values
                    weights = [1.0 / (i + 1) for i in range(n_cand)]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0.0
                    x = valid_candidates[-1]
                    for i in range(n_cand):
                        cum += weights[i]
                        if r <= cum:
                            x = valid_candidates[i]
                            break

                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows

    # Helper: local search / swap improvement
    def local_search(rows, k, max_iters=200):
        """
        Try to reduce the maximum entry by adjusting values in each row.
        Uses a multi-pass approach: repeatedly try to shrink the largest entry
        of each row (sorted descending) until no improvement is found.
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
            
            # Process rows in order of their current maximum entry (largest first)
            row_order = sorted(range(len(rows)), key=lambda i: rows[i][-1], reverse=True)
            
            any_progress = False
            for max_idx in row_order:
                if _time_left() < 2.0:
                    break
                row = rows[max_idx]
                max_val = row[-1]

                # Try to reduce the last element
                found = False
                for new_val in range(row[-2] + 1, max_val):
                    # Check if replacing last element with new_val keeps all diffs valid
                    old_diffs = {row[-1] - a for a in row[:-1]}
                    new_diffs = {new_val - a for a in row[:-1]}
                    if new_diffs & (used - old_diffs):
                        continue
                    if new_val <= row[-2]:
                        continue
                    # Valid! Apply
                    used -= old_diffs
                    used |= new_diffs
                    row[-1] = new_val
                    found = True
                    improved = True
                    any_progress = True
                    break

                if not found:
                    # Try to reduce second-to-last element, then re-optimize last
                    if len(row) >= 3:
                        pen_val = row[-2]
                        for new_pen in range(row[-3] + 1, pen_val):
                            # Remove diffs involving pen_val and last
                            old_diffs = set()
                            for a in row[:-2]:
                                old_diffs.add(pen_val - a)
                                old_diffs.add(row[-1] - a)
                            old_diffs.add(row[-1] - pen_val)
                            
                            # Check new pen_val doesn't conflict
                            new_pen_diffs = {new_pen - a for a in row[:-2]}
                            if new_pen_diffs & (used - old_diffs):
                                continue
                            if new_pen <= row[-3]:
                                continue
                            
                            # Try to find a valid last value >= row[-2] + 1
                            # Actually last must be > new_pen
                            best_last = None
                            for new_last in range(new_pen + 1, row[-1] + 1):
                                new_last_diffs = {new_last - a for a in row[:-2]}
                                new_last_diffs.add(new_last - new_pen)
                                all_new = new_pen_diffs | new_last_diffs
                                if all_new & (used - old_diffs):
                                    continue
                                if best_last is None or new_last < best_last:
                                    best_last = new_last
                            
                            if best_last is not None:
                                used -= old_diffs
                                new_pen_diffs = {new_pen - a for a in row[:-2]}
                                new_last_diffs = {best_last - a for a in row[:-2]}
                                new_last_diffs.add(best_last - new_pen)
                                used |= new_pen_diffs
                                used |= new_last_diffs
                                row[-2] = new_pen
                                row[-1] = best_last
                                found = True
                                improved = True
                                any_progress = True
                                break

            if not any_progress:
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

    # Apply local search to baseline first
    if _time_left() > 5.0:
        improved = local_search(best_rows, k, max_iters=200)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    # Try randomized greedy and lookahead greedy multiple times
    num_attempts = 80
    for attempt in range(num_attempts):
        if _time_left() < 8.0:
            break
        # Alternate between strategies
        if attempt % 3 == 0:
            rows = randomized_greedy(n_rows, k, rng)
        elif attempt % 3 == 1:
            rows = lookahead_greedy(n_rows, k, rng)
        else:
            rows = randomized_greedy(n_rows, k, rng)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
            # Try local search on the new best
            if _time_left() > 8.0:
                improved = local_search(best_rows, k, max_iters=150)
                if improved:
                    new_max = max(max(r) for r in improved)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = improved

    # Final intensive local search on best
    if _time_left() > 5.0:
        improved = local_search(best_rows, k, max_iters=500)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    # One more pass with even more iterations if time allows
    if _time_left() > 3.0:
        improved = local_search(best_rows, k, max_iters=300)
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
