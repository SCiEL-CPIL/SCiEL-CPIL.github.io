"""Difference triangle set (7,5) solver with randomized restarts and local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_scope = None
    best_rows = None

    # Total number of differences per row: k*(k+1)//2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def is_valid(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_complete(partial_row, used, k):
        """Try to complete a partial row greedily. Returns completed row or None."""
        row = list(partial_row)
        while len(row) < k + 1:
            # Try smallest x that doesn't conflict
            x = row[-1] + 1
            found = False
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used = used | new_diffs
                    found = True
                    break
                x += 1
                # Safety: if x gets too large, give up
                if x > 200:
                    return None
        return row

    def random_row(used, k, max_val):
        """Generate a random valid row given used differences."""
        # Start with 0
        row = [0]
        while len(row) < k + 1:
            # Try random values in increasing order
            candidates = list(range(row[-1] + 1, max_val + 1))
            random.shuffle(candidates)
            placed = False
            for x in candidates:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used = used | new_diffs
                    placed = True
                    break
            if not placed:
                return None
        return row

    def local_search(rows, used, k, max_iter=5000):
        """Try to reduce the maximum entry by swapping out large values."""
        current_rows = [list(r) for r in rows]
        current_scope = max(max(r) for r in current_rows)
        best_scope_ls = current_scope
        best_rows_ls = [list(r) for r in current_rows]

        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Pick a random row and try to reduce its last element
            ri = random.randrange(n_rows)
            row = current_rows[ri]
            if len(row) < k + 1:
                continue

            # Remove the last element
            last = row[-1]
            row = row[:-1]

            # Recompute used without this row's differences
            used_new = set()
            for r in current_rows:
                if r is not row:  # This won't work correctly; let's recompute properly
                    pass

            # Actually, let's just recompute used from all other rows
            used_other = set()
            for idx, r in enumerate(current_rows):
                if idx != ri:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            used_other.add(r[j] - r[i])

            # Now try to find a new last element smaller than last
            # We need k elements total, we have k-1 (indices 0 to k-2), need one more
            # Wait, row has k elements now (after removing last from k+1). We need 1 more.
            # Actually row started with k+1, removed last -> k elements. Need 1 more to make k+1.
            # The new element must be > row[-1] (current last of the k elements)
            # But we want to reduce the max, so we want new_last < last.
            # However new_last must be > row[k-1] (the previous second-to-last).
            # Actually the row is increasing, so new element must be > row[-1].
            # If row[-1] >= last, we can't reduce. But row[-1] was the second-to-last, so row[-1] < last.
            # We need new_x such that row[-1] < new_x < last, and all diffs are valid.

            start = row[-1] + 1
            end = last  # exclusive
            if start >= end:
                # Can't reduce
                current_rows[ri] = row + [last]
                continue

            # Try to find a valid x in [start, end)
            found_x = None
            for x in range(start, end):
                new_diffs = {x - a for a in row}
                if not new_diffs & used_other:
                    found_x = x
                    break

            if found_x is not None:
                new_row = row + [found_x]
                new_scope = max(max(r) for r in current_rows[:ri] + [new_row] + current_rows[ri+1:])
                if new_scope < best_scope_ls:
                    best_scope_ls = new_scope
                    best_rows_ls = current_rows[:ri] + [new_row] + current_rows[ri+1:]
                current_rows[ri] = new_row
            else:
                # Restore
                current_rows[ri] = row + [last]

        return best_rows_ls, best_scope_ls

    # Strategy 1: Greedy with different orderings
    for attempt in range(10):
        if time.time() > deadline:
            break

        used = set()
        rows = []
        success = True

        for i in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                found = False
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                    x += 1
                    if x > 200:
                        success = False
                        break
                if not success:
                    break
            if not success:
                break
            rows.append(row)

        if success and len(rows) == n_rows:
            scope = max(max(r) for r in rows)
            if best_scope is None or scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    # Strategy 2: Randomized construction with restarts
    for attempt in range(200):
        if time.time() > deadline:
            break

        random.seed(attempt)
        used = set()
        rows = []
        success = True

        for i in range(n_rows):
            # Generate a random row
            row = [0]
            while len(row) < k + 1:
                # Try random candidates
                placed = False
                max_try = 100
                for _ in range(max_try):
                    # Pick a random x > row[-1]
                    x = row[-1] + 1 + random.randrange(0, 50)
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            if not success:
                break
            rows.append(row)

        if success and len(rows) == n_rows:
            scope = max(max(r) for r in rows)
            if best_scope is None or scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    # Strategy 3: Local search improvement on best found
    if best_rows is not None:
        for attempt in range(20):
            if time.time() > deadline:
                break

            random.seed(1000 + attempt)
            # Compute used from best_rows
            used = set()
            for r in best_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        used.add(r[j] - r[i])

            improved_rows, improved_scope = local_search(best_rows, used, k, max_iter=2000)
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = [list(r) for r in improved_rows]

    # Final safety: if we didn't find anything, return a trivial valid set (may have large scope)
    if best_rows is None:
        # Fallback: greedy without time limit check (should be fast for small inputs)
        used = set()
        rows = []
        for i in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        best_rows = rows

    return best_rows
# EVOLVE-BLOCK-END
