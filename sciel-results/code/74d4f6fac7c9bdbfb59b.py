"""Improved difference-triangle constructor with deadline guard and smarter search."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    # n_rows rows, each with k+1 entries starting at 0, strictly increasing.
    # All positive differences within a row must be distinct across all rows.
    # We want to minimize the largest entry.

    # For (7,5): 7 rows of 6 integers each. Total differences per row = C(6,2)=15.
    # Total distinct differences needed = 7*15 = 105.
    # Counting bound: m >= 105 (since differences range 1..m).
    # Best known: 112.

    # Strategy:
    # 1. Greedy construction with backtracking to find a valid set.
    # 2. Try to minimize the maximum entry via local search / iterative improvement.
    # 3. Use randomized restarts with different orderings if time permits.

    best_rows = None
    best_max = float('inf')

    def greedy_construct(n_rows, k, max_val, seed=0):
        """Greedy: fill rows one by one, picking smallest available values."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for _ in range(k):
                found = False
                x = 1
                while x <= max_val:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                    x += 1
                if not found:
                    return None
            rows.append(row)
        return rows

    def backtrack_construct(n_rows, k, max_val, seed=0):
        """Backtracking search to find a valid set with max entry <= max_val."""
        rng = random.Random(seed)
        used = set()
        rows = []

        def solve(row_idx, col_idx, current_row):
            if time.time() > deadline:
                return False
            if row_idx == n_rows:
                return True
            if col_idx == k + 1:
                rows.append(list(current_row))
                return solve(row_idx + 1, 0, [0])

            start = current_row[-1] + 1 if col_idx > 0 else 1
            for x in range(start, max_val + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used:
                    continue
                # Check if this leads to a dead end (simple pruning)
                used |= new_diffs
                current_row.append(x)
                if solve(row_idx, col_idx + 1, current_row):
                    return True
                current_row.pop()
                used -= new_diffs

            return False

        if solve(0, 0, [0]):
            return rows
        return None

    def local_improve(rows, max_val, seed=0):
        """Try to reduce the maximum entry via local search."""
        rng = random.Random(seed)
        current = [list(r) for r in rows]
        current_max = max(max(r) for r in current)

        for _ in range(5000):
            if time.time() > deadline:
                break
            # Pick a random row and try to reduce its largest element
            r = rng.randrange(len(current))
            row = current[r]
            if len(row) < 2:
                continue
            largest_idx = len(row) - 1
            old_val = row[largest_idx]
            if old_val <= 1:
                continue

            # Try replacing the largest value with a smaller one
            used_all = set()
            for rr in current:
                for i in range(len(rr)):
                    for j in range(i+1, len(rr)):
                        used_all.add(rr[j] - rr[i])

            # Remove this row's diffs
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_all.discard(row[j] - row[i])

            # Try smaller values for the last element
            for new_val in range(old_val - 1, row[-2], -1):
                new_diffs = {new_val - a for a in row[:-1]}
                if not (new_diffs & used_all):
                    row[largest_idx] = new_val
                    current_max = max(current_max, new_val)
                    # Recompute actual max
                    current_max = max(max(rr) for rr in current)
                    break

        return current, current_max

    # Phase 1: Try greedy with increasing max_val
    # Counting bound for (7,5) is 105, best known 112
    lower_bound = 105
    upper_bound = 130  # start high, then tighten

    # First, try to find any valid solution
    found = False
    for mv in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break
        rows = greedy_construct(n_rows, k, mv, seed=42)
        if rows is not None:
            found = True
            best_rows = rows
            best_max = mv
            break

    if not found:
        # Try backtracking from the lower bound
        for mv in range(lower_bound, upper_bound + 1):
            if time.time() > deadline:
                break
            rows = backtrack_construct(n_rows, k, mv, seed=42)
            if rows is not None:
                found = True
                best_rows = rows
                best_max = mv
                break

    if not found:
        # Last resort: greedy with high max
        rows = greedy_construct(n_rows, k, 200, seed=42)
        if rows is not None:
            best_rows = rows
            best_max = max(max(r) for r in rows)
        else:
            # Fallback: just return greedy with very high max
            rows = greedy_construct(n_rows, k, 500, seed=42)
            if rows is not None:
                best_rows = rows
                best_max = max(max(r) for r in rows)
            else:
                # Absolute fallback: return a trivial valid set
                # (may not be minimal but is valid)
                used = set()
                rows = []
                for r in range(n_rows):
                    row = [0]
                    x = 1
                    while len(row) < k + 1:
                        new = {x - a for a in row}
                        if not (new & used):
                            row.append(x)
                            used |= new
                        x += 1
                    rows.append(row)
                best_rows = rows
                best_max = max(max(r) for r in rows)

    # Phase 2: Local improvement to reduce max
    if time.time() < deadline - 10:
        improved, new_max = local_improve(best_rows, best_max, seed=42)
        if new_max < best_max:
            best_rows = improved
            best_max = new_max

    # Phase 3: Try randomized restarts if time remains
    if time.time() < deadline - 20:
        for seed in range(100):
            if time.time() > deadline - 5:
                break
            # Try greedy with a slightly randomized order
            rows = greedy_construct(n_rows, k, best_max, seed=seed + 100)
            if rows is not None:
                cur_max = max(max(r) for r in rows)
                if cur_max < best_max:
                    best_rows = rows
                    best_max = cur_max
                    # Try to improve further
                    if time.time() < deadline - 10:
                        improved, new_max = local_improve(best_rows, best_max, seed=seed)
                        if new_max < best_max:
                            best_rows = improved
                            best_max = new_max

    return best_rows
# EVOLVE-BLOCK-END
