<<<<<<< SEARCH
def _polish_scope(solution, n_rows, k, deadline):
    """
    Aggressive polishing: repeatedly try single-element shifts to minimize scope.
    This is a tighter local search that can find small improvements.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    # Iteratively try to reduce each element
    for iteration in range(20):
        if time.time() > deadline - 2.0:
            break

        any_improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Try shifting down by 1, 2, ... up to original - row[pos-1] - 1
                for new_val in range(original - 1, row[pos - 1], -1):
                    # Check if new value creates conflicts
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    # Check if new_diffs conflict with (used_diffs - old_diffs)
                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    # Check if new_diffs are all positive
                    if any(d <= 0 for d in new_diffs):
                        continue

                    # Apply the change
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_improved = True
                    break

        if not any_improved:
            break

    return best
=======
def _polish_scope(solution, n_rows, k, deadline):
    """
    Aggressive polishing: repeatedly try single-element shifts to minimize scope.
    This is a tighter local search that can find small improvements.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    # Iteratively try to reduce each element - more aggressive with multiple passes
    for iteration in range(50):
        if time.time() > deadline - 2.0:
            break

        any_improved = False
        # Randomize row order to escape local minima
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        for row_idx in row_order:
            row = best[row_idx]
            # Randomize position order too
            pos_order = list(range(k_plus_1 - 1, 0, -1))
            random.shuffle(pos_order)
            for pos in pos_order:
                original = row[pos]
                # Try shifting down by 1, 2, ... up to original - row[pos-1] - 1
                for new_val in range(original - 1, row[pos - 1], -1):
                    # Check if new value creates conflicts
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    # Check if new_diffs conflict with (used_diffs - old_diffs)
                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    # Check if new_diffs are all positive
                    if any(d <= 0 for d in new_diffs):
                        continue

                    # Apply the change
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_improved = True
                    break

        if not any_improved:
            break

    return best
>>>>>>> REPLACE
