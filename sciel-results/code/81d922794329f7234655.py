"""Difference triangle set solver: Efficient SA with diff-counting + row reconstruction."""
import time
import sys
import random
import math

sys.setrecursionlimit(100000)


def _validate(rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _greedy(n_rows, k):
    """Fast greedy: smallest next mark with no difference collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            ok = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows


def _get_diffs(row):
    """Return list of all pairwise differences in a row."""
    diffs = []
    L = len(row)
    for i in range(L):
        for j in range(i + 1, L):
            diffs.append(row[j] - row[i])
    return diffs


def _reconstruct_row(r_idx, rows, used_diffs, upper_bound, row_len):
    """Re-greedy row r_idx against the current set of used differences from other rows."""
    current_row = rows[r_idx]
    current_diffs = set(_get_diffs(current_row))
    for d in current_diffs:
        used_diffs.discard(d)

    new_row = [0]
    x = 0
    while len(new_row) < row_len:
        x += 1
        if x > upper_bound:
            break
        new_diffs = {x - a for a in new_row}
        if not (new_diffs & used_diffs):
            new_row.append(x)
            used_diffs |= new_diffs

    if len(new_row) == row_len:
        rows[r_idx] = new_row
        return True
    else:
        # Rollback
        for d in set(_get_diffs(new_row)):
            used_diffs.discard(d)
        used_diffs |= current_diffs
        return False


def _hill_climb_rows(rows, time_limit, row_len, n_rows):
    """Iteratively reconstruct rows to shrink the bounding box."""
    rows = [r[:] for r in rows]
    max_val = max(max(r) for r in rows)
    start_time = time.time()

    # Build used_diffs set
    used_diffs = set()
    for r in rows:
        for d in _get_diffs(r):
            used_diffs.add(d)

    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for r_idx in range(n_rows):
            if time.time() - start_time >= time_limit:
                break
            old_max = rows[r_idx][-1]
            # Try with current global max as bound
            if _reconstruct_row(r_idx, rows, used_diffs, max_val, row_len):
                new_max = rows[r_idx][-1]
                if new_max < old_max:
                    improved = True
                    new_global = max(max(r) for r in rows)
                    if new_global < max_val:
                        max_val = new_global

    return rows, max(max(r) for r in rows)


def _sa_optimize(n_rows, k, initial_rows, deadline, seed=42):
    """Simulated Annealing with efficient diff-counting for single-entry moves."""
    rng = random.Random(seed)
    rows = [list(r) for r in initial_rows]
    L = k + 1  # entries per row

    # Initialize diff counts
    row_diffs = [_get_diffs(r) for r in rows]
    diff_counts = {}
    for rd in row_diffs:
        for d in rd:
            diff_counts[d] = diff_counts.get(d, 0) + 1

    def compute_cost():
        collisions = 0
        for d, c in diff_counts.items():
            if c > 1:
                collisions += c - 1
        max_val = max(r[-1] for r in rows)
        return collisions * 100000 + max_val

    current_cost = compute_cost()
    best_cost = current_cost
    best_rows = [r[:] for r in rows]
    best_max = max(r[-1] for r in best_rows)

    T = 50.0
    T_min = 0.1
    cooling = 0.9997

    while time.time() < deadline:
        if T < T_min:
            break

        r_idx = rng.randrange(n_rows)
        i_idx = rng.randrange(1, L)

        row = rows[r_idx]
        old_val = row[i_idx]

        lower = row[i_idx - 1] + 1
        upper = row[i_idx + 1] - 1 if i_idx < L - 1 else best_max + 30
        upper = min(upper, best_max + 50)

        if lower > upper:
            T *= cooling
            continue

        new_val = rng.randint(lower, upper)
        if new_val == old_val:
            T *= cooling
            continue

        # Remove old diffs for this row
        old_diffs = row_diffs[r_idx]
        for d in old_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

        # Apply new value
        row[i_idx] = new_val
        new_diffs = _get_diffs(row)
        row_diffs[r_idx] = new_diffs

        # Add new diffs
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1

        new_cost = compute_cost()
        delta_c = new_cost - current_cost

        if delta_c < 0 or (delta_c < 1000 and rng.random() < math.exp(-delta_c / T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [r[:] for r in rows]
                best_max = max(r[-1] for r in best_rows)
        else:
            # Revert
            row[i_idx] = old_val
            for d in new_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            row_diffs[r_idx] = old_diffs

        T *= cooling

    return best_rows


def _row_reconstruct_search(rows, deadline, row_len, n_rows):
    """Use iterative row reconstruction with randomization to improve solution."""
    rows = [r[:] for r in rows]
    max_val = max(max(r) for r in rows)
    start_time = time.time()
    rng = random.Random(12345)

    # Build used_diffs set
    used_diffs = set()
    for r in rows:
        for d in _get_diffs(r):
            used_diffs.add(d)

    best_rows = [r[:] for r in rows]
    best_max = max_val

    for iteration in range(200):
        if time.time() > deadline:
            break

        # Pick a random row to reconstruct
        r_idx = rng.randrange(n_rows)
        old_row = rows[r_idx][:]
        old_max = rows[r_idx][-1]

        # Try reconstruction with progressively tighter bounds
        bound = min(max_val, old_max + 5)
        if _reconstruct_row(r_idx, rows, used_diffs, bound, row_len):
            new_max = rows[r_idx][-1]
            if new_max < old_max:
                new_global = max(max(r) for r in rows)
                if new_global < best_max:
                    best_max = new_global
                    best_rows = [r[:] for r in rows]
                    max_val = new_global
        else:
            # Rollback was done inside _reconstruct_row, but let's verify
            # Actually _reconstruct_row handles rollback internally
            pass

        # Occasionally try a different strategy: reconstruct with slightly relaxed bound
        if iteration % 10 == 5:
            r_idx = rng.randrange(n_rows)
            bound = max_val + 10
            if _reconstruct_row(r_idx, rows, used_diffs, bound, row_len):
                new_global = max(max(r) for r in rows)
                if new_global < best_max:
                    best_max = new_global
                    best_rows = [r[:] for r in rows]
                    max_val = new_global

    return best_rows, best_max


def construct(n_rows, k):
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s
    row_len = k + 1

    # Phase 1: Greedy baseline
    greedy_rows = _greedy(n_rows, k)
    best = greedy_rows
    best_scope = _scope(best)

    # Phase 2: Hill climbing with row reconstruction
    if time.time() < deadline - 30.0:
        try:
            hc_rows, hc_scope = _hill_climb_rows(greedy_rows, min(30.0, deadline - time.time() - 5.0), row_len, n_rows)
            if _validate(hc_rows, k) and hc_scope < best_scope:
                best = hc_rows
                best_scope = hc_scope
        except Exception:
            pass

    # Phase 3: SA with efficient diff counting
    if time.time() < deadline - 20.0:
        try:
            sa_deadline = min(deadline - 5.0, time.time() + 40.0)
            sa_rows = _sa_optimize(n_rows, k, best, sa_deadline, seed=42)
            if sa_rows is not None and _validate(sa_rows, k):
                sa_scope = _scope(sa_rows)
                if sa_scope < best_scope:
                    best = sa_rows
                    best_scope = sa_scope
        except Exception:
            pass

    # Phase 4: Row reconstruction search on best
    if time.time() < deadline - 10.0:
        try:
            rr_deadline = min(deadline - 5.0, time.time() + 30.0)
            rr_rows, rr_scope = _row_reconstruct_search(best, rr_deadline, row_len, n_rows)
            if _validate(rr_rows, k) and rr_scope < best_scope:
                best = rr_rows
                best_scope = rr_scope
        except Exception:
            pass

    # Phase 5: Final SA polish
    if time.time() < deadline - 8.0:
        try:
            sa2_deadline = min(deadline - 3.0, time.time() + 20.0)
            sa2_rows = _sa_optimize(n_rows, k, best, sa2_deadline, seed=999)
            if sa2_rows is not None and _validate(sa2_rows, k):
                sa2_scope = _scope(sa2_rows)
                if sa2_scope < best_scope:
                    best = sa2_rows
                    best_scope = sa2_scope
        except Exception:
            pass

    # Final validation
    if not _validate(best, k):
        best = _greedy(n_rows, k)

    return best
