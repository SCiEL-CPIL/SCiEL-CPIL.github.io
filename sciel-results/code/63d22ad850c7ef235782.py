import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin before 120s limit

    # Generate initial solution using greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)

    # Iterated Local Search with valid perturbations
    no_improve_count = 0
    iter_count = 0
    while time.time() < deadline - 20:
        iter_count += 1
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]

        # Perform several random valid moves (perturbation)
        n_moves = random.randint(3, 8)
        for _ in range(n_moves):
            move_type = random.random()
            if move_type < 0.6:
                # Single position change
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
                if new_val is not None and new_val != candidate[r_idx][pos]:
                    candidate[r_idx][pos] = new_val
            elif move_type < 0.8:
                # Swap two positions within a row
                r_idx = random.randint(0, n_rows - 1)
                if k >= 2:
                    p1, p2 = random.sample(range(1, k + 1), 2)
                    if _try_swap(candidate, r_idx, p1, p2, n_rows, k):
                        pass  # swap applied
            else:
                # Rebuild one row greedily
                if n_rows > 1:
                    r_idx = random.randint(0, n_rows - 1)
                    other_rows = [r for i, r in enumerate(candidate) if i != r_idx]
                    used_diffs_other = _compute_all_diffs(other_rows)
                    new_row = _greedy_row_given_diffs(used_diffs_other, k)
                    if new_row is not None:
                        candidate[r_idx] = new_row

        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline)

        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # If we've had many iterations without improvement, try a stronger perturbation
        if no_improve_count >= 4:
            no_improve_count = 0
            # Try rebuilding 1-2 rows
            n_rebuild = random.randint(1, min(2, n_rows))
            for _ in range(n_rebuild):
                if time.time() > deadline - 20:
                    break
                r_idx = random.randint(0, n_rows - 1)
                other_rows = [r for i, r in enumerate(candidate) if i != r_idx]
                used_diffs_other = _compute_all_diffs(other_rows)
                new_row = _greedy_row_given_diffs(used_diffs_other, k)
                if new_row is not None:
                    candidate[r_idx] = new_row
            improved2 = _local_search_minimize(candidate, n_rows, k, deadline)
            if improved2:
                curr_score2 = max(max(r) for r in improved2)
                if curr_score2 < best_score:
                    best_score = curr_score2
                    best_solution = improved2

    # Final polish pass with remaining time
    if time.time() < deadline - 3:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution


def _greedy_row_given_diffs(used_diffs, k):
    """Build a single row greedily given already-used differences."""
    row = [0]
    current_max = 0
    while len(row) < k + 1:
        found = False
        for x in range(current_max + 1, current_max + 2000):
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
    return row

def _try_swap(rows, r_idx, p1, p2, n_rows, k):
    """Try swapping values at positions p1 and p2 in row r_idx. Returns True if successful."""
    row = rows[r_idx]
    v1, v2 = row[p1], row[p2]
    if v1 == v2:
        return False
    # After swap: row[p1]=v2, row[p2]=v1
    # Check that new ordering is still strictly increasing
    new_row = row[:]
    new_row[p1] = v2
    new_row[p2] = v1
    for i in range(len(new_row) - 1):
        if new_row[i] >= new_row[i + 1]:
            return False

    # Compute all diffs, remove old diffs involving p1 or p2, add new ones
    all_diffs = _compute_all_diffs(rows)

    # Old diffs involving p1 or p2
    old_involving = set()
    for i in range(len(row)):
        if i == p1 or i == p2:
            continue
        old_involving.add(abs(row[i] - v1))
        old_involving.add(abs(row[i] - v2))
    old_involving.add(abs(v1 - v2))

    forbidden = all_diffs - old_involving

    # New diffs involving p1 (now v2) or p2 (now v1)
    new_involving = set()
    for i in range(len(row)):
        if i == p1 or i == p2:
            continue
        new_involving.add(abs(new_row[i] - v2))
        new_involving.add(abs(new_row[i] - v1))
    new_involving.add(abs(v1 - v2))

    # Check no new diff collides with forbidden
    if new_involving & forbidden:
        return False

    # Check no internal collision among new_involving
    # (they're all different by construction since values are distinct)

    # Apply swap
    row[p1] = v2
    row[p2] = v1
    return True

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

    # Compute all diffs once
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
    current_max = max(max(r) for r in rows)
    upper_bound = min(current_max + 30, 250)
    if pos + 1 < len(row):
        upper_bound = min(upper_bound, row[pos + 1] - 1)

    if upper_bound < lower_bound:
        return None

    # Collect valid candidates, bias toward lower values
    valid_candidates = []
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
            valid_candidates.append(new_val)

    if not valid_candidates:
        return None

    # Weighted random choice favoring lower values
    if len(valid_candidates) <= 5:
        return random.choice(valid_candidates)

    max_candidate = max(valid_candidates)
    weights = []
    for v in valid_candidates:
        weight = (max_candidate - v + 1) ** 2
        weights.append(weight)

    total_weight = sum(weights)
    r = random.uniform(0, total_weight)
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return valid_candidates[i]
    return valid_candidates[-1]

def _local_search_minimize(rows, n_rows, k, deadline):
    """Greedy local search to minimize max entry with incremental diff tracking."""
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)

    # Precompute per-row diff sets for faster updates
    row_diffs = [_row_diffs(best_rows[i]) for i in range(n_rows)]

    while time.time() < deadline - 5:
        progress = False

        # Randomize row order for diversity
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 5:
                break

            row = best_rows[r_idx]
            rd = row_diffs[r_idx]

            # Process positions right-to-left
            for pos in range(k, 0, -1):
                old_val = row[pos]

                # Compute diffs involving old_val at this position
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
                    # Update used_diffs and row_diffs incrementally
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                        rd.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                        rd.add(d)
                    row[pos] = found
                    progress = True

        if not progress:
            break

    return best_rows

def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction with randomization for diversity."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            search_start = current_max + 1
            search_end = current_max + 1500
            for x in range(search_start, search_end):
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
