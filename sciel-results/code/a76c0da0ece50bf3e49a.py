"""Difference triangle set (4,7): SA with fast conflict counting, binding-row-first."""
import time
import random

_DEADLINE = 145.0  # safety margin under 150s wall-clock


def _row_diffs(row):
    """All positive differences a[j]-a[i] for i<j in one row."""
    d = []
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.append(row[j] - ai)
    return d


def _conflict_count(rows):
    """Count duplicate differences across all rows (0 = valid)."""
    seen = set()
    conflicts = 0
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                conflicts += 1
            else:
                seen.add(d)
    return conflicts


def _row_scope(row):
    return row[-1]


def _total_conflicts(rows):
    return _conflict_count(rows)


def _greedy_row(used, k):
    """Build one row greedily given already-used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row


def _greedy_all(n_rows, k):
    """Initial greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k))
        used |= set(_row_diffs(rows[-1]))
    return rows


def _valid(rows):
    """True if all positive differences across rows are distinct."""
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _sa_optimize(rows, k, deadline, rng):
    """Simulated annealing: mutate one element of one row, accept by conflict delta.
    Prioritize the row with the largest scope (binding row) for mutation."""
    n = len(rows)
    best_rows = [list(r) for r in rows]
    best_conf = _total_conflicts(best_rows)
    best_scope = max(_row_scope(r) for r in best_rows)

    cur_rows = [list(r) for r in rows]
    cur_conf = _total_conflicts(cur_rows)

    # Temperature schedule
    t_start = max(1.0, cur_conf * 0.5)
    t_end = 0.01
    # We'll run until deadline, adapting temperature by elapsed fraction

    t0 = time.time()
    elapsed_total = deadline - t0
    if elapsed_total <= 1:
        return best_rows

    # Precompute per-row difference multisets for fast delta
    # We'll recompute conflicts incrementally: keep a global counter of diff occurrences.
    from collections import Counter

    def build_counter(rows):
        c = Counter()
        for row in rows:
            for d in _row_diffs(row):
                c[d] += 1
        return c

    def conflicts_from_counter(c):
        return sum(v - 1 for v in c.values() if v > 1)

    cur_counter = build_counter(cur_rows)
    cur_conf = conflicts_from_counter(cur_counter)

    # Track best
    if cur_conf < best_conf or (cur_conf == best_conf and max(_row_scope(r) for r in cur_rows) < best_scope):
        best_conf = cur_conf
        best_rows = [list(r) for r in cur_rows]
        best_scope = max(_row_scope(r) for r in best_rows)

    # Choose row to mutate: with probability proportional to scope (binding row first)
    # and to current conflict contribution.
    def pick_row():
        # Weight each row by (1 + its scope) so the largest row is more often picked
        weights = [1.0 + _row_scope(r) for r in cur_rows]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i
        return n - 1

    # Temperature decay over time
    steps = 0
    while time.time() < deadline:
        elapsed = time.time() - t0
        frac = min(1.0, elapsed / elapsed_total)
        # Exponential-ish cooling
        temp = t_end + (t_start - t_end) * (1.0 - frac) ** 1.5
        if temp < 1e-6:
            temp = 1e-6

        ri = pick_row()
        row = cur_rows[ri]
        # Pick an index to mutate (not index 0, which is fixed at 0)
        idx = 1 + rng.randrange(len(row) - 1)
        old_val = row[idx]

        # Decide delta: move within a window
        # Window size adapts to temperature
        window = int(1 + temp * 5)
        delta = rng.randint(-window, window)
        new_val = old_val + delta
        if new_val <= 0:
            new_val = old_val + 1
        # Keep strictly increasing: must be > row[idx-1] and < row[idx+1] (if exists)
        lo = row[idx - 1] + 1
        hi = (row[idx + 1] - 1) if idx + 1 < len(row) else 10**9
        if new_val < lo:
            new_val = lo
        if new_val > hi:
            new_val = hi
        if new_val == old_val:
            continue

        # Compute conflict delta for this row only (differences involving idx)
        # Old diffs involving idx:
        old_d = []
        for j in range(len(row)):
            if j != idx:
                if j < idx:
                    old_d.append(old_val - row[j])
                else:
                    old_d.append(row[j] - old_val)
        # New diffs:
        new_d = []
        for j in range(len(row)):
            if j != idx:
                if j < idx:
                    new_d.append(new_val - row[j])
                else:
                    new_d.append(row[j] - new_val)

        # Delta in global conflict count:
        # For each diff d, going from count c to c-1 changes conflicts by:
        #   if c>=2: (c-2) - (c-1) ... let's just compute contribution change.
        # conflicts contribution of a diff with count c is max(0, c-1).
        # Removing an occurrence: c -> c-1, delta_conf = max(0,c-1) - max(0,c-2)
        #   = (c-1) - (c-2) if c>=2 else 0 = 1 if c>=2 else 0.
        # Adding an occurrence: c -> c+1, delta_conf = max(0,c) - max(0,c-1)
        #   = c - (c-1) if c>=1 else 0 = 1 if c>=1 else 0.

        delta_conf = 0
        # Remove old diffs
        for d in old_d:
            c = cur_counter[d]
            # removing one: new count c-1
            if c >= 2:
                delta_conf -= 1  # conflict count decreases by 1
            cur_counter[d] = c - 1
        # Add new diffs
        for d in new_d:
            c = cur_counter[d]
            # adding one: new count c+1
            if c >= 1:
                delta_conf += 1  # conflict count increases by 1
            cur_counter[d] = c + 1

        new_conf = cur_conf + delta_conf

        # Accept?
        if new_conf < cur_conf:
            accept = True
        elif new_conf == cur_conf:
            # Accept if scope didn't increase, or with small probability
            new_scope = _row_scope(row) if False else max(row)  # row not yet updated
            # Compute new scope of this row after change
            # row[idx] would become new_val
            new_row_scope = max(row[:idx] + [new_val] + row[idx+1:])
            old_row_scope = _row_scope(row)
            # total scope change
            other_max = max(_row_scope(r) for i, r in enumerate(cur_rows) if i != ri) if n > 1 else 0
            new_total_scope = max(new_row_scope, other_max)
            old_total_scope = max(old_row_scope, other_max)
            if new_total_scope <= old_total_scope:
                accept = True
            else:
                accept = rng.random() < 0.3
        else:
            # Worse conflicts: accept with probability exp(-delta/temp)
            delta = new_conf - cur_conf
            if delta > 0:
                # Avoid overflow
                if delta / temp > 50:
                    accept = False
                else:
                    accept = rng.random() < (2.718281828 ** (-(delta / temp)))
            else:
                accept = True

        if accept:
            row[idx] = new_val
            cur_conf = new_conf
            # Update best
            if cur_conf < best_conf or (cur_conf == best_conf and max(_row_scope(r) for r in cur_rows) < best_scope):
                best_conf = cur_conf
                best_rows = [list(r) for r in cur_rows]
                best_scope = max(_row_scope(r) for r in best_rows)
        else:
            # Revert counter
            for d in new_d:
                cur_counter[d] -= 1
            for d in old_d:
                cur_counter[d] += 1

        steps += 1
        # Periodic local polish on best: try to reduce scope of binding row
        if steps % 5000 == 0:
            _local_polish(best_rows, k, rng)
            bc = _total_conflicts(best_rows)
            if bc < best_conf:
                best_conf = bc
                best_scope = max(_row_scope(r) for r in best_rows)

    return best_rows


def _local_polish(rows, k, rng):
    """Try to reduce the scope of the largest row by small local moves, keeping validity."""
    # Find the row with max scope
    scopes = [_row_scope(r) for r in rows]
    bi = scopes.index(max(scopes))
    row = rows[bi]
    improved = True
    while improved:
        improved = False
        for idx in range(1, len(row)):
            # Try to decrease row[idx] by 1, 2, ... as long as strictly increasing and no new conflicts
            for dec in range(1, 6):
                new_val = row[idx] - dec
                if new_val <= row[idx - 1]:
                    break
                if idx + 1 < len(row) and new_val >= row[idx + 1]:
                    continue
                # Check conflicts
                # Remove old diffs involving idx, add new
                old_d = []
                for j in range(len(row)):
                    if j != idx:
                        old_d.append(abs(row[j] - row[idx]))
                new_d = []
                for j in range(len(row)):
                    if j != idx:
                        new_d.append(abs(row[j] - new_val))
                # All diffs in all rows
                all_diffs = []
                for r in rows:
                    all_diffs.extend(_row_diffs(r))
                # Remove old_d occurrences
                from collections import Counter
                c = Counter(all_diffs)
                for d in old_d:
                    c[d] -= 1
                for d in new_d:
                    c[d] += 1
                if all(v <= 1 for v in c.values()):
                    # Valid! Apply
                    row[idx] = new_val
                    improved = True
                    break
    # Re-sort safety (shouldn't be needed)
    row.sort()


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all positive differences distinct across rows."""
    rng = random.Random(12345)

    # Greedy initial
    rows = _greedy_all(n_rows, k)

    t0 = time.time()
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 5:
        return rows

    # SA optimization
    rows = _sa_optimize(rows, k, _DEADLINE, rng)

    # Final local polish on the result
    _local_polish(rows, k, rng)

    # Ensure validity; if SA somehow broke it, fall back to greedy
    if not _valid(rows):
        rows = _greedy_all(n_rows, k)

    return rows
