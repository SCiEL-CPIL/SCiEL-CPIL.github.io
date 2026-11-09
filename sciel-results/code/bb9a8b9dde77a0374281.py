"""Backtracking solver for difference triangle set (n_rows, k) with
conflict-directed pruning, randomized restarts, and scope-bounded search."""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 132.0  # safety margin before 150s kill

    # Precompute: number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    best_rows = None
    best_scope = float('inf')

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_scope(rows):
        return max(max(row) for row in rows)

    # Counting bound: minimum scope such that there are enough distinct
    # positive differences in [1, scope] to assign to all rows.
    # Differences available in [1, S]: S (values 1..S)
    # But we need total_diffs distinct values, each >= 1, so S >= total_diffs.
    # This is a weak bound but useful for pruning.
    counting_bound = total_diffs  # minimum possible scope

    # Backtracking search for a single row given used differences
    def build_row_backtrack(row, used, scope_limit, rng, depth=0):
        """Recursively build a row starting from 'row' (list, starts with [0]),
        trying to keep all entries <= scope_limit. Returns True on success."""
        if len(row) == k + 1:
            return True

        current_max = row[-1]
        # Candidate range: current_max+1 to scope_limit
        # Use randomization: shuffle candidates or pick random window
        start = current_max + 1
        end = scope_limit

        if start > end:
            return False

        # Generate candidates with some randomization
        # Strategy: try a random subset of candidates in [start, end]
        # For efficiency, limit candidates per level
        max_candidates = 40
        if end - start + 1 <= max_candidates:
            candidates = list(range(start, end + 1))
            rng.shuffle(candidates)
        else:
            # Pick random candidates spread across the range
            candidates = set()
            # Always include the smallest few (greedy bias)
            for c in range(start, min(start + 5, end + 1)):
                candidates.add(c)
            # Add random candidates
            while len(candidates) < max_candidates:
                c = rng.randint(start, end)
                candidates.add(c)
            candidates = list(candidates)
            rng.shuffle(candidates)

        for candidate in candidates:
            # Check if candidate creates valid differences
            new_diffs = {candidate - a for a in row}
            if new_diffs & used:
                continue
            # Check all new diffs are within scope
            if max(new_diffs) > scope_limit:
                continue

            row.append(candidate)
            used.update(new_diffs)

            if build_row_backtrack(row, used, scope_limit, rng, depth + 1):
                return True

            row.pop()
            used.difference_update(new_diffs)

        return False

    # Full backtracking: build all rows one by one
    def backtrack_all_rows(used, rows, scope_limit, rng):
        """Recursively build remaining rows."""
        if len(rows) == n_rows:
            return True

        row = [0]
        if build_row_backtrack(row, used, scope_limit, rng):
            rows.append(row)
            if backtrack_all_rows(used, rows, scope_limit, rng):
                return True
            rows.pop()
            # Remove this row's differences from used
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.discard(row[j] - row[i])
        return False

    # Local search: try to reduce scope by adjusting entries
    def local_search(rows, deadline):
        best = [list(r) for r in rows]
        best_scope = compute_scope(best)
        improved = True

        while improved and time.time() < deadline - 10:
            improved = False
            # Find the row with the maximum entry
            max_idx = max(range(n_rows), key=lambda i: max(best[i]))
            row = best[max_idx]
            last_val = row[-1]

            if last_val <= best_scope:
                continue

            # Compute differences used by other rows
            used_by_others = set()
            for i in range(n_rows):
                if i == max_idx:
                    continue
                for j in range(len(best[i])):
                    for l in range(j + 1, len(best[i])):
                        used_by_others.add(best[i][l] - best[i][j])

            # Differences within the row except those involving the last element
            used_by_row_no_last = set()
            for i in range(len(row) - 1):
                for j in range(i + 1, len(row) - 1):
                    used_by_row_no_last.add(row[j] - row[i])

            # Try to reduce the last element
            for new_val in range(last_val - 1, row[-2], -1):
                new_diffs = {new_val - a for a in row[:-1]}
                if new_diffs & used_by_others:
                    continue
                if new_diffs & used_by_row_no_last:
                    continue
                if new_val <= row[-2]:
                    continue

                row[-1] = new_val
                if is_valid(best):
                    new_scope = compute_scope(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        break
                else:
                    row[-1] = last_val
                    break

            # Also try reducing non-last entries (more expensive)
            if not improved:
                for pos in range(1, len(row)):
                    for new_val in range(row[pos] - 1, row[pos - 1], -1):
                        if new_val <= row[pos - 1]:
                            continue
                        # Check all differences involving this position
                        old_val = row[pos]
                        row[pos] = new_val
                        if is_valid(best):
                            new_scope = compute_scope(best)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                improved = True
                                break
                        row[pos] = old_val
                        if improved:
                            break
                    if improved:
                        break

        return best, best_scope

    # Pure greedy fallback
    def pure_greedy(n_rows, k):
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
        return rows

    # Phase 0: Get a baseline with pure greedy
    baseline = pure_greedy(n_rows, k)
    best_rows = baseline
    best_scope = compute_scope(baseline)

    # Phase 1: Backtracking search with decreasing scope limits
    # Start from current best scope and try to find solutions with smaller scope
    rng = random.Random(42)

    # Try scopes from best_scope-1 down to counting_bound
    # But limit the number of attempts based on time
    max_attempts = 500
    attempts = 0

    for scope in range(best_scope - 1, max(counting_bound - 1, 0), -1):
        if time.time() > deadline - 30 or attempts >= max_attempts:
            break

        # Try multiple random restarts for this scope
        restarts_for_scope = min(20, max(1, (deadline - time.time() - 30) // 5))
        for restart in range(restarts_for_scope):
            if time.time() > deadline - 30:
                break
            if attempts >= max_attempts:
                break

            attempts += 1
            rng_local = random.Random(hash((scope, restart)) % (2**32))
            used = set()
            rows = []
            if backtrack_all_rows(used, rows, scope, rng_local):
                if is_valid(rows):
                    s = compute_scope(rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [list(r) for r in rows]
                        # Found a better solution; try to improve further
                        break

        # If we found a solution at this scope, we can stop trying this scope
        # (since we're going downward, any solution at scope is valid)
        # But we might find an even better one with a different restart
        # So continue to next scope

    # Phase 2: Local search on the best solution
    if best_rows is not None and time.time() < deadline - 15:
        best_rows, best_scope = local_search(best_rows, deadline)

    # Phase 3: More aggressive backtracking with randomized row order
    if best_rows is not None and time.time() < deadline - 20:
        # Try different row construction orders
        for trial in range(100):
            if time.time() > deadline - 15:
                break
            # Randomize the scope slightly around current best
            scope = best_scope - random.randint(1, max(1, best_scope // 10))
            if scope < counting_bound:
                scope = counting_bound

            rng_local = random.Random(hash((trial, scope)) % (2**32))
            used = set()
            rows = []
            if backtrack_all_rows(used, rows, scope, rng_local):
                if is_valid(rows):
                    s = compute_scope(rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [list(r) for r in rows]
                        if time.time() < deadline - 15:
                            best_rows, best_scope = local_search(best_rows, deadline)

    # Final fallback
    if best_rows is None:
        best_rows = pure_greedy(n_rows, k)

    return best_rows
