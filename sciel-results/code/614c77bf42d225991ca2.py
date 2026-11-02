import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin before 120s limit

    # Generate initial solution using fast greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)

    # Iterated Local Search with valid perturbations
    while time.time() < deadline - 15:
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]

        # Perform several random valid moves (perturbation)
        n_moves = random.randint(3, 10)
        for _ in range(n_moves):
            r_idx = random.randint(0, n_rows - 1)

            # With some probability, replace an entire row
            if random.random() < 0.25:
                new_row = _replace_row(candidate, r_idx, n_rows, k)
                if new_row is not None:
                    candidate[r_idx] = new_row
                continue

            pos = random.randint(1, k)  # Don't touch position 0

            # Find a random valid new value for candidate[r_idx][pos]
            new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val

        # Two-stage local search: first unbounded (restructure), then minimize
        improved = _local_search_unbounded(candidate, n_rows, k, deadline - 10)
        if improved:
            improved = _local_search_minimize(improved, n_rows, k, deadline)

        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved

    # Final polish pass with remaining time
    if time.time() < deadline - 2:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution

def _row_diffs(row):
    """Return the set of all positive differences within a single row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s

def _compute_all_diffs(rows):
    """Return the union of all positive differences across all rows."""
    all_diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                all_diffs.add(r[j] - ri)
    return all_diffs

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """Find a random valid value for rows[r_idx][pos] different from current."""
    row = rows[r_idx]
    old_val = row[pos]

    # Build the set of differences not involving the old value at this position
    all_diffs = _compute_all_diffs(rows)

    diffs_involving_old = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)

    forbidden_diffs = all_diffs - diffs_involving_old

    # Search for a new value
    lower_bound = row[pos - 1] + 1
    upper_bound = 300  # Slightly wider than before
    if pos + 1 < len(row):
        upper_bound = min(upper_bound, row[pos + 1] - 1)

    if upper_bound < lower_bound:
        return None

    # Collect valid candidates
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue

        # Check validity: each new diff must not collide with forbidden_diffs
        # or with another new diff in the same row.
        valid = True
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val

            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)

        if valid:
            valid_candidates.append(new_val)

    if not valid_candidates:
        return None

    # Pick randomly
    return random.choice(valid_candidates)

def _replace_row(rows, r_idx, n_rows, k):
    """Replace an entire row with a new valid row."""
    # Compute all diffs excluding the row to be replaced
    used_diffs = _compute_all_diffs(rows)
    old_row = rows[r_idx]
    for d in _row_diffs(old_row):
        used_diffs.discard(d)

    # Try to build a new row avoiding used_diffs
    row = [0]
    current_max = 0
    for _ in range(k):
        found = False
        for x in range(current_max + 1, current_max + 500):
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                current_max = x
                found = True
                break
        if not found:
            return None

    return row

def _local_search_minimize(rows, n_rows, k, deadline):
    """Greedy local search to minimize max entry.

    Walks positions right-to-left within each row so that lowering an earlier
    position does not invalidate later positions (they are already fixed and
    their diffs are accounted for in used_diffs).
    """
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)

    while time.time() < deadline - 5:
        progress = False

        # Randomize row order for diversity
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 5:
                break

            row = best_rows[r_idx]

            # Process positions right-to-left
            for pos in range(k, 0, -1):
                old_val = row[pos]

                # Compute forbidden diffs: all used diffs minus those involving old_val
                diffs_involving_old = set()
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)

                forbidden_diffs = used_diffs - diffs_involving_old

                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < len(row):
                    upper_bound = min(upper_bound, row[pos + 1] - 1)

                if upper_bound < lower_bound:
                    continue

                # Search for smallest valid value
                found = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val

                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                    if valid:
                        found = new_val
                        break

                if found is not None and found < old_val:
                    # Update used_diffs incrementally
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True

        if not progress:
            break

    return best_rows

def _local_search_unbounded(rows, n_rows, k, deadline):
    """Local search that can move values up or down to restructure rows.

    This helps escape local minima where the current upper bound prevents
    finding a better configuration.
    """
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)
    best_score = max(max(r) for r in best_rows)

    while time.time() < deadline - 5:
        progress = False

        # Randomize row order for diversity
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 5:
                break

            row = best_rows[r_idx]

            # Process positions right-to-left
            for pos in range(k, 0, -1):
                old_val = row[pos]

                # Compute forbidden diffs: all used diffs minus those involving old_val
                diffs_involving_old = set()
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)

                forbidden_diffs = used_diffs - diffs_involving_old

                lower_bound = row[pos - 1] + 1
                # Allow searching above current value to restructure
                upper_bound = old_val + 20
                if pos + 1 < len(row):
                    upper_bound = min(upper_bound, row[pos + 1] - 1)

                if upper_bound < lower_bound:
                    continue

                # Search for a valid value that improves or maintains the score
                found = None
                for new_val in range(lower_bound, upper_bound + 1):
                    if new_val == old_val:
                        continue
                    valid = True
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val

                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                    if valid:
                        # Only accept if it doesn't increase the max
                        if new_val <= best_score:
                            found = new_val
                            break

                if found is not None and found != old_val:
                    # Update used_diffs incrementally
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True

        if not progress:
            break

    return best_rows

def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows
