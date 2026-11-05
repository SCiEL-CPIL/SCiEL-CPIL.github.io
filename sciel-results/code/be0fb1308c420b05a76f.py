"""Combined: column-wise list scheduling with bitmask conflict checking + local search."""
import time
import random


def _deadline_guard(start, margin=15.0, limit=150.0):
    """Return True if we should stop (elapsed + margin >= limit)."""
    return (time.time() - start) + margin >= limit


def _score_of(rows, k):
    """Largest entry across all rows."""
    return max(max(r) for r in rows) if rows else 0


def _is_valid(rows, k):
    """Check that all positive differences within rows are distinct across rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def construct(n_rows, k):
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    # ---- Phase 1: Column-wise list scheduling with bitmask ----
    # used_mask: integer bitmask, bit d set iff difference d is used
    # rows: list of lists, each starting with [0]

    rows = [[0] for _ in range(n_rows)]
    used_mask = 0

    for col in range(k):
        if _deadline_guard(start, 10.0, 150.0):
            break

        # Sort row indices by their current last element (ascending)
        row_order = sorted(range(n_rows), key=lambda r: rows[r][-1])

        for r in row_order:
            # Find smallest x > rows[r][-1] such that no new diff conflicts
            x = rows[r][-1] + 1
            while True:
                # Compute new differences bitmask
                new_mask = 0
                conflict = False
                for a in rows[r]:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                if not conflict:
                    break
                x += 1

            # Commit
            for a in rows[r]:
                d = x - a
                used_mask |= (1 << d)
            rows[r].append(x)

    best_score = _score_of(rows, k)
    best_rows = [list(r) for r in rows]

    if _deadline_guard(start, 8.0, 150.0):
        return best_rows

    # ---- Phase 2: Local search to reduce the maximum entry ----
    # Rebuild individual rows greedily with bitmask, trying to minimize max.
    # Use cost heuristic: prefer values that use larger differences (less scarce).

    random.seed(42)
    improved = True
    while improved and not _deadline_guard(start, 8.0, 150.0):
        improved = False
        for r in random.sample(range(n_rows), n_rows):
            if _deadline_guard(start, 8.0, 150.0):
                break

            # Remove this row's differences from used_mask
            old_row = best_rows[r]
            row_mask = 0
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    d = old_row[j] - old_row[i]
                    row_mask |= (1 << d)
            used_mask_without = used_mask & ~row_mask

            # Rebuild this row greedily, trying to minimize the max
            # Target: keep max below best_score if possible
            target_scope = best_score - 1  # try to beat current best
            if target_scope < 0:
                target_scope = 0

            new_row = [0]
            for _ in range(k):
                x = new_row[-1] + 1
                best_x = None
                best_cost = float('inf')
                # Search for feasible x, prefer ones with lower cost (larger diffs)
                # Limit search range to keep it fast
                search_limit = min(x + 200, target_scope + 1)
                while x <= search_limit:
                    new_mask = 0
                    conflict = False
                    for a in new_row:
                        d = x - a
                        bit = 1 << d
                        if used_mask_without & bit:
                            conflict = True
                            break
                        new_mask |= bit
                    if not conflict:
                        # Cost: sum of 1/d for new diffs (prefer large diffs)
                        cost = 0.0
                        for a in new_row:
                            d = x - a
                            cost += 1.0 / d
                        if cost < best_cost:
                            best_cost = cost
                            best_x = x
                            best_new_mask = new_mask
                    x += 1

                if best_x is None:
                    # No feasible x found in range; break and keep old row
                    break

                # Commit
                for a in new_row:
                    d = best_x - a
                    used_mask_without |= (1 << d)
                new_row.append(best_x)

            # Check if new row is complete and better
            if len(new_row) == k + 1:
                new_max = new_row[-1]
                old_max = old_row[-1]
                if new_max < old_max:
                    # Update used_mask
                    new_row_mask = 0
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            new_row_mask |= (1 << d)
                    used_mask = used_mask_without | new_row_mask
                    best_rows[r] = new_row
                    best_score = _score_of(best_rows, k)
                    improved = True
                else:
                    # Restore original
                    used_mask = used_mask_without | row_mask
            else:
                # Incomplete row, restore
                used_mask = used_mask_without | row_mask

    # ---- Phase 3: Try different row orderings for column-wise build ----
    if not _deadline_guard(start, 5.0, 150.0):
        for trial in range(30):
            if _deadline_guard(start, 5.0, 150.0):
                break

            rows2 = [[0] for _ in range(n_rows)]
            used2 = 0

            for col in range(k):
                if _deadline_guard(start, 5.0, 150.0):
                    break
                # Random permutation of row order for this column
                row_order = list(range(n_rows))
                random.shuffle(row_order)

                for r in row_order:
                    x = rows2[r][-1] + 1
                    while True:
                        new_mask = 0
                        conflict = False
                        for a in rows2[r]:
                            d = x - a
                            bit = 1 << d
                            if used2 & bit:
                                conflict = True
                                break
                            new_mask |= bit
                        if not conflict:
                            break
                        x += 1
                    for a in rows2[r]:
                        d = x - a
                        used2 |= (1 << d)
                    rows2[r].append(x)

            trial_score = _score_of(rows2, k)
            if trial_score < best_score:
                best_score = trial_score
                best_rows = [list(r) for r in rows2]

    # ---- Phase 4: Final local search pass with tighter target ----
    if not _deadline_guard(start, 5.0, 150.0):
        for _pass in range(5):
            if _deadline_guard(start, 5.0, 150.0):
                break
            improved = False
            for r in random.sample(range(n_rows), n_rows):
                if _deadline_guard(start, 5.0, 150.0):
                    break
                old_row = best_rows[r]
                row_mask = 0
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        d = old_row[j] - old_row[i]
                        row_mask |= (1 << d)
                used_mask_without = used_mask & ~row_mask

                new_row = [0]
                for _ in range(k):
                    x = new_row[-1] + 1
                    best_x = None
                    best_cost = float('inf')
                    search_limit = min(x + 150, best_score)
                    while x <= search_limit:
                        new_mask = 0
                        conflict = False
                        for a in new_row:
                            d = x - a
                            bit = 1 << d
                            if used_mask_without & bit:
                                conflict = True
                                break
                            new_mask |= bit
                        if not conflict:
                            cost = 0.0
                            for a in new_row:
                                d = x - a
                                cost += 1.0 / d
                            if cost < best_cost:
                                best_cost = cost
                                best_x = x
                                best_new_mask = new_mask
                        x += 1

                    if best_x is None:
                        break
                    for a in new_row:
                        d = best_x - a
                        used_mask_without |= (1 << d)
                    new_row.append(best_x)

                if len(new_row) == k + 1:
                    new_max = new_row[-1]
                    old_max = old_row[-1]
                    if new_max < old_max:
                        new_row_mask = 0
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                d = new_row[j] - new_row[i]
                                new_row_mask |= (1 << d)
                        used_mask = used_mask_without | new_row_mask
                        best_rows[r] = new_row
                        best_score = _score_of(best_rows, k)
                        improved = True
                    else:
                        used_mask = used_mask_without | row_mask
                else:
                    used_mask = used_mask_without | row_mask

    return best_rows
