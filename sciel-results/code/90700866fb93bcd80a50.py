"""Difference triangle set (7,5) – construct(n_rows, k)."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Return a list of n_rows rows, each a list of k+1 strictly increasing ints
    starting at 0, such that all positive differences within a row are distinct
    across all rows. Minimizes the largest entry.
    """
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s kill

    # Total number of differences per row: k*(k+1)/2
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row

    # Best solution found so far
    best_scope = float('inf')
    best_rows = None

    # Greedy baseline (deterministic)
    def greedy_deterministic():
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

    best_rows = greedy_deterministic()
    best_scope = max(max(r) for r in best_rows)

    # Check time
    if time.time() > deadline:
        return best_rows

    # Randomized greedy with multiple restarts and different row orderings
    def randomized_greedy(rng, max_scope_hint=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if max_scope_hint is not None and x > max_scope_hint:
                    return None
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Try randomized restarts with various seeds
    rng = random.Random(42)
    attempts = 0
    max_attempts = 500

    while time.time() < deadline - 5.0 and attempts < max_attempts:
        attempts += 1
        # Occasionally try with a scope hint slightly below current best
        if best_scope < float('inf') and rng.random() < 0.3:
            hint = best_scope - 1
        else:
            hint = None
        result = randomized_greedy(rng, hint)
        if result is not None:
            scope = max(max(r) for r in result)
            if scope < best_scope:
                best_scope = scope
                best_rows = result
                # If we hit the counting bound, we can't do better
                # Counting bound: total_diffs <= scope*(scope+1)/2
                # So scope must be at least the smallest m with m*(m+1)/2 >= total_diffs
                # For (7,5): total_diffs = 7*15 = 105, m=14 gives 105, so bound is 14? 
                # Wait, that's the bound on the number of differences, not the scope.
                # The simple counting bound mentioned is 105 for the largest entry.
                # Let's not stop early unless we're very close.
                if scope <= 106:
                    break

    # Local improvement: try to swap or adjust entries to reduce scope
    if best_rows is not None and time.time() < deadline - 10.0:
        # Try to reduce the largest entry by removing it and finding a replacement
        # This is a heuristic local search
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            current_rows = [list(r) for r in best_rows]
            current_scope = max(max(r) for r in current_rows)
            
            # Find the row with the largest last element
            max_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
            max_val = current_rows[max_row_idx][-1]
            
            if max_val <= 105:
                break
            
            # Try to replace the last element with a smaller one
            row = current_rows[max_row_idx]
            # Compute all differences in this row except those involving the last element
            diffs_without_last = set()
            for i in range(len(row) - 1):
                for j in range(i + 1, len(row) - 1):
                    diffs_without_last.add(row[j] - row[i])
            
            # All differences from other rows
            other_diffs = set()
            for ri in range(n_rows):
                if ri == max_row_idx:
                    continue
                r = current_rows[ri]
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        other_diffs.add(r[j] - r[i])
            
            forbidden = diffs_without_last | other_diffs
            
            # Try smaller values for the last element
            found = False
            for new_last in range(row[-2] + 1, max_val):
                new_diffs = {new_last - a for a in row[:-1]}
                if not (new_diffs & forbidden):
                    row[-1] = new_last
                    # Update best
                    new_scope = max(max(r) for r in current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                        improved = True
                    found = True
                    break
            
            if not found:
                # Try another row
                # Shuffle which row we try to improve
                pass

    return best_rows
# EVOLVE-BLOCK-END
