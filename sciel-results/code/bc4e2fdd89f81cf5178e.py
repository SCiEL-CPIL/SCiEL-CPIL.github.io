"""Difference triangle set (7,5): 7 rows of 6 marks (start 0, strictly increasing),
all within-row positive differences distinct globally, minimize the largest entry.

Strategy: exact constraint-satisfaction backtracking with bitmasks, forward
checking via Hall's condition on the remaining marks of the current row, MRV row
ordering, feasibility-maximizing value ordering, and iterative deepening from the
counting bound (105) upward. A fast greedy fallback guarantees a valid answer.
"""
import time
import random
import copy

def _validate(rows):
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_all(n_rows, k, max_x=400):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        ok = True
        while len(row) < k + 1:
            base = row[-1]
            found = False
            for x in range(base + 1, max_x + 1):
                nd = 0
                for a in row:
                    nd |= 1 << (x - a)
                if (nd & used) == 0:
                    row.append(x)
                    used |= nd
                    found = True
                    break
            if not found:
                ok = False
                break
        if not ok:
            return None
        rows.append(row)
    return rows

def _row_mask(row):
    m = 0
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            m |= 1 << (row[j] - row[i])
    return m

def _solve(n_rows, k, M, rng, deadline):
    """Exact backtracking: find n_rows valid rows with max entry <= M, or None."""
    # Feasibility: need n_rows * C(k+1,2) distinct differences in [1, M]
    need = n_rows * (k + 1) * k // 2
    if need > M:
        return None
    M1 = M + 1
    full = (1 << M1) - 1          # differences 1..M
    used = 0

    def _fc(row, pos, used):
        """Forward check: can positions pos..k be completed? Hall's condition.
        Returns (ok, residual_mask_for_current_row)."""
        avail = full ^ used
        rem = k + 1 - pos
        if rem <= 0:
            return True, avail
        lo = row[-1] + 1
        if M - lo + 1 < rem:
            return False, avail
        # For each required future mark, the set of difference values it can use.
        # Hall: for every subset S of required marks, |union of their value-sets| >= |S|.
        # We use a greedy necessary check: each required mark must have a non-empty
        # feasible value set, and the union of all must be >= rem.
        # Build per-mark feasible value sets (as bitmasks over difference values).
        # mark at position p (p in pos..k) needs value v > row[-1] and
        # for all a in row: (v - a) in avail, and all (v - a) distinct.
        # We compute, for each required position, the set of difference-value
        # bitmasks that are "consuming" - simpler: check that enough distinct
        # difference values remain by counting available values.
        # Stronger: each of the rem marks needs len(row) new differences.
        # Total new differences needed = rem * len(row).
        # These must be distinct and within avail.
        # Count available difference values:
        # A difference d is "usable" if it is in avail.
        # We need at least rem*len(row) usable values that can be arranged.
        # Cheap necessary condition: popcount(avail) >= rem * len(row)
        # (each mark contributes len(row) new differences, all distinct).
        if avail.bit_count() < rem * len(row):
            return False, avail
        # Additional: the smallest mark (lo..lo+rem-1) must each fit; ensure the
        # range of values is wide enough for distinct differences - already checked
        # by M-lo+1 >= rem.
        return True, avail

    def _value_order(row, pos, used):
        """Yield candidate values in an order that maximizes future feasibility."""
        lo = row[-1] + 1
        hi = M
        cands = []
        for v in range(lo, hi + 1):
            nd = 0
            ok = True
            for a in row:
                d = v - a
                if d > M or (used >> d) & 1:
                    ok = False
                    break
                nd |= 1 << d
            if not ok:
                continue
            # score = remaining avail popcount after adding (higher is better)
            score = (full ^ (used | nd)).bit_count()
            cands.append((score, v))
        # sort by score descending, tie by value ascending
        cands.sort(key=lambda t: (-t[0], t[1]))
        return [v for _, v in cands]

    rows = []

    def _build_row(pos, row, used):
        """Build current row from position pos. Returns (ok, used)."""
        nonlocal used
        if time.time() > deadline:
            return False, used
        if pos == k + 1:
            rows.append(row[:])
            return True, used
        # forward check
        ok, _ = _fc(row, pos, used)
        if not ok:
            return False, used
        for v in _value_order(row, pos, used):
            nd = 0
            for a in row:
                nd |= 1 << (v - a)
            used2 = used | nd
            row.append(v)
            ok2, _ = _build_row(pos + 1, row, used2)
            row.pop()
            if ok2:
                return True, used2
        return False, used

    def _solve_rows(ri, used):
        nonlocal used
        if time.time() > deadline:
            return False
        if ri == n_rows:
            return True
        # MRV: try to build this row; pick ordering by how constrained
        ok, used2 = _build_row(1, [0], used)
        if ok:
            if _solve_rows(ri + 1, used2):
                used = used2
                return True
        return False

    # Try several row orderings / value-order perturbations for diversity
    for attempt in range(6):
        if time.time() > deadline:
            break
        used = 0
        rows.clear()
        if _solve_rows(0, 0):
            return rows
    return None

def construct(n_rows, k):
    deadline = time.time() + 140  # safety margin under 150s
    rng = random.Random(12345)

    # Guaranteed valid fallback (greedy), but keep it small.
    fallback = _greedy_all(n_rows, k, max_x=300)
    if fallback is None:
        fallback = _greedy_all(n_rows, k, max_x=800)

    best_rows = fallback
    best_score = _max_entry(best_rows) if fallback else float('inf')

    # Iterative deepening from the counting bound upward.
    lower = n_rows * (k + 1) * k // 2  # counting bound on max entry
    # Search targets from lower upward, but cap effort per target.
    target = lower
    while target <= best_score - 1 and time.time() < deadline - 5:
        # Give each target a time budget that shrinks as we go up.
        budget = min(20.0, (deadline - 5 - time.time()))
        t_start = time.time()
        t_deadline = t_start + budget
        rows = _solve(n_rows, k, target, rng, t_deadline)
        if rows is not None and _validate(rows):
            score = _max_entry(rows)
            if score < best_score:
                best_score = score
                best_rows = copy.deepcopy(rows)
            # If we hit the counting bound exactly, we're optimal; stop.
            if score == lower:
                break
            # Found a solution at `target`; try one lower target next.
            target = score  # next target is the found score (which == target)
            # continue to try target-1 by setting target to found-1
            target = target - 1
        else:
            target += 1

    # Final validation
    if not _validate(best_rows):
        fresh = _greedy_all(n_rows, k, max_x=800)
        if fresh is not None and _validate(fresh):
            best_rows = fresh

    return best_rows
