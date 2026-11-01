"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.

Strategy: greedy multi-start + conflict-directed value removal (CDVR) local search
on a complete assignment, plus targeted descent on the global maximum.
"""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 138.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


def _row_diffs(row):
    """List of all positive differences in a row, indexed by (i, j) with i<j."""
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs


def _validate_solution(rows, n_rows, k):
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k):
            if row[i] >= row[i + 1]:
                return False
        d = set()
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                dv = row[j] - row[i]
                if dv in d:
                    return False
                d.add(dv)
        if d & seen:
            return False
        seen |= d
    return True


def _greedy_build(n_rows, k, row_order, rng):
    """Build rows in given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows


def _greedy_interleaved(n_rows, k, rng, strategy="smallest"):
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        candidates = [r for r in range(n_rows) if len(rows[r]) < k + 1]
        if not candidates:
            break
        if strategy == "smallest":
            best_r = min(candidates, key=lambda r: rows[r][-1])
        elif strategy == "largest":
            best_r = max(candidates, key=lambda r: rows[r][-1])
        elif strategy == "random":
            best_r = rng.choice(candidates)
        elif strategy == "balanced":
            best_r = max(candidates, key=lambda r: len(rows[r]))
        else:
            best_r = min(candidates, key=lambda r: rows[r][-1])
        row = rows[best_r]
        x = row[-1] + 1
        while True:
            ok = True
            for a in row:
                if (x - a) in used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
                placed += 1
                break
            x += 1
    return rows


def _try_greedy_variants(n_rows, k, rng, num_trials=30):
    best_rows = None
    best_max = float('inf')
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    strategies = ["smallest", "largest", "random", "balanced"]
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        rows = _greedy_interleaved(n_rows, k, rng, strategy=strategies[t % len(strategies)])
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    return best_rows, best_max


def _targeted_descent(rows, n_rows, k, time_budget=30.0):
    """Deterministic descent: repeatedly reduce the largest entries."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    row_diffs = [set(_row_diffs(rows[r])) for r in range(n_rows)]

    def other_diffs(exclude):
        s = set()
        for r in range(n_rows):
            if r != exclude:
                s |= row_diffs[r]
        return s

    while time.time() - start < time_budget and _time_left() > 3.0:
        max_val = max(max(rows[r]) for r in range(n_rows))
        if max_val < best_max:
            best_max = max_val
            best_rows = [list(r) for r in rows]
        max_rows = [r for r in range(n_rows) if max(rows[r]) == max_val]
        improved = False
        for r in max_rows:
            if _time_left() < 3.0:
                break
            row = rows[r]
            old_val = row[-1]
            for new_val in range(old_val - 1, row[-2], -1):
                base = set()
                for i in range(k):
                    for j in range(i + 1, k):
                        base.add(row[j] - row[i])
                ok = True
                for a in row[:-1]:
                    d = new_val - a
                    if d in base:
                        ok = False
                        break
                if not ok:
                    continue
                od = other_diffs(r)
                for a in row[:-1]:
                    if (new_val - a) in od:
                        ok = False
                        break
                if not ok:
                    continue
                row[-1] = new_val
                row_diffs[r] = set(_row_diffs(row))
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best_rows, best_max


class _ConflictState:
    """
    Complete assignment with a conflict graph.
    Each difference value maps to the list of (row, pos) pairs that produce it,
    where pos is the index of the element that, together with a smaller element,
    creates the difference. We track collisions: a value is 'conflicted' if it
    appears more than once across all rows.
    """

    def __init__(self, rows, n_rows, k):
        self.rows = [list(r) for r in rows]
        self.n_rows = n_rows
        self.k = k
        self._rebuild()

    def _rebuild(self):
        # diff_owner: value -> list of (row, pos) where pos is the larger index
        self.diff_owner = {}
        for r in range(self.n_rows):
            row = self.rows[r]
            for i in range(self.k + 1):
                for j in range(i + 1, self.k + 1):
                    d = row[j] - row[i]
                    if d not in self.diff_owner:
                        self.diff_owner[d] = []
                    self.diff_owner[d].append((r, j))
        # collision_count per (row, pos): number of values that this (row,pos)
        # participates in which are conflicted
        self.pos_conflict = [0] * self.n_rows
        self.total_collisions = 0
        for d, owners in self.diff_owner.items():
            if len(owners) > 1:
                self.total_collisions += len(owners) - 1
                for (r, j) in owners:
                    self.pos_conflict[r] += 1
        self.scope = max(max(r) for r in self.rows)

    def _diffs_for_pos(self, r, pos):
        """Return list of (value, i) where i is the smaller index, for element at pos."""
        row = self.rows[r]
        return [(row[pos] - row[i], i) for i in range(pos)]

    def _is_valid_value(self, r, pos, val, exclude_self=True):
        """Check if setting rows[r][pos]=val is internally and externally valid."""
        row = self.rows[r]
        if pos > 0 and val <= row[pos - 1]:
            return False
        if pos < self.k and val >= row[pos + 1]:
            return False
        # internal: differences with other elements in same row must be unique
        internal = set()
        for i in range(pos):
            d = val - row[i]
            if d in internal:
                return False
            internal.add(d)
        # also check differences with elements after pos
        for j in range(pos + 1, self.k + 1):
            d = row[j] - val
            if d in internal:
                return False
            internal.add(d)
        # external: no difference can appear in another row
        for d in internal:
            owners = self.diff_owner.get(d, [])
            for (orow, opos) in owners:
                if orow != r:
                    return False
                # if orow == r, it's our own diff; fine as long as it's the same pair
        return True

    def _remove_pos(self, r, pos):
        """Remove element at (r, pos) from diff_owner. Returns list of (value, i) removed."""
        row = self.rows[r]
        removed = []
        for i in range(pos):
            d = row[pos] - row[i]
            lst = self.diff_owner[d]
            lst.remove((r, pos))
            removed.append((d, i))
        for j in range(pos + 1, self.k + 1):
            d = row[j] - row[pos]
            lst = self.diff_owner[d]
            lst.remove((r, j))
            removed.append((d, j))
        return removed

    def _add_pos(self, r, pos, val, removed_info):
        """Add element at (r, pos) with value val, using removed_info to restore."""
        self.rows[r][pos] = val
        for (d, other_idx) in removed_info:
            if d not in self.diff_owner:
                self.diff_owner[d] = []
            # Determine which index is the larger one
            if other_idx < pos:
                self.diff_owner[d].append((r, pos))
            else:
                self.diff_owner[d].append((r, other_idx))

    def _update_conflict_counts(self, r, pos, old_val):
        """Update pos_conflict after changing rows[r][pos] from old_val to new."""
        # We'll just recompute the conflict counts for simplicity if needed,
        # but for efficiency we do incremental updates.
        pass

    def cdvr_step(self, rng):
        """
        One CDVR step: find the (row, pos) with the most conflicts, remove its value,
        and try to reassign it to the smallest valid alternative.
        Returns True if a change was made.
        """
        if self.total_collisions == 0:
            return False
        # Find the position with the most conflicts
        best_r, best_pos, best_conf = -1, -1, -1
        for r in range(self.n_rows):
            if self.pos_conflict[r] > best_conf:
                best_conf = self.pos_conflict[r]
                best_r = r
                # pick a random pos in this row for diversity
                best_pos = rng.randrange(1, self.k + 1)
        if best_r < 0:
            return False

        old_val = self.rows[best_r][best_pos]
        removed_info = self._remove_pos(best_r, best_pos)

        # Try candidate values: old_val +/- small deltas, and values near neighbors
        candidates = set()
        row = self.rows[best_r]
        lower = row[best_pos - 1] + 1 if best_pos > 0 else 1
        upper = row[best_pos + 1] - 1 if best_pos < self.k else old_val + 50
        for d in range(-10, 11):
            candidates.add(old_val + d)
        # Also try values that are just above lower or just below upper
        candidates.add(lower)
        candidates.add(upper)
        candidates = sorted([v for v in candidates if lower <= v <= upper and v != old_val])

        # Also try the old value (in case removal+re-add helps conflict structure)
        # But we want to change, so skip old_val.

        for new_val in candidates:
            if not self._is_valid_value(best_r, best_pos, new_val, exclude_self=True):
                continue
            # Tentatively add
            self._add_pos(best_r, best_pos, new_val, removed_info)
            # Recompute conflict counts (expensive but correct)
            old_total = self.total_collisions
            self._rebuild_conflicts_only()
            if self.total_collisions < old_total:
                return True
            # If no improvement, revert
            self._remove_pos(best_r, best_pos)
            self._add_pos(best_r, best_pos, old_val, removed_info)
            self._rebuild_conflicts_only()
            # Try next candidate
            # But to avoid infinite loop, break after trying a few
            break
        # If no candidate worked, restore old value
        self._add_pos(best_r, best_pos, old_val, removed_info)
        self._rebuild_conflicts_only()
        return False

    def _rebuild_conflicts_only(self):
        """Recompute pos_conflict and total_collisions from diff_owner."""
        self.pos_conflict = [0] * self.n_rows
        self.total_collisions = 0
        for d, owners in self.diff_owner.items():
            if len(owners) > 1:
                self.total_collisions += len(owners) - 1
                for (r, j) in owners:
                    self.pos_conflict[r] += 1
        self.scope = max(max(r) for r in self.rows)


def _local_search_cdvr(rows, n_rows, k, rng, time_budget=30.0):
    """CDVR local search: minimize collisions first, then scope."""
    state = _ConflictState(rows, n_rows, k)
    best_rows = [list(r) for r in state.rows]
    best_scope = state.scope
    best_collisions = state.total_collisions
    start = time.time()
    no_improve = 0
    max_no_improve = 200

    while time.time() - start < time_budget and _time_left() > 5.0:
        made_change = state.cdvr_step(rng)
        if made_change:
            no_improve = 0
            if state.scope < best_scope or (state.scope == best_scope and state.total_collisions < best_collisions):
                best_scope = state.scope
                best_collisions = state.total_collisions
                best_rows = [list(r) for r in state.rows]
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Random kick: change a random element
                r = rng.randrange(n_rows)
                pos = rng.randrange(1, k + 1)
                old_val = state.rows[r][pos]
                removed_info = state._remove_pos(r, pos)
                # Try a few random values
                found = False
                for _ in range(20):
                    new_val = old_val + rng.randint(-15, 15)
                    if new_val == old_val:
                        continue
                    if state._is_valid_value(r, pos, new_val, exclude_self=True):
                        state._add_pos(r, pos, new_val, removed_info)
                        state._rebuild_conflicts_only()
                        found = True
                        break
                if not found:
                    state._add_pos(r, pos, old_val, removed_info)
                    state._rebuild_conflicts_only()
                no_improve = 0
    return best_rows, best_scope


def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)

    # Phase 1: Greedy multi-start
    best_rows, best_max = _try_greedy_variants(n_rows, k, rng, num_trials=40)
    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)

    # Phase 2: Targeted descent
    if _time_left() > 15.0:
        ls_rows, ls_max = _targeted_descent(best_rows, n_rows, k, time_budget=min(20.0, _time_left() - 10.0))
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    # Phase 3: CDVR local search
    if _time_left() > 20.0:
        cdvr_rows, cdvr_max = _local_search_cdvr(best_rows, n_rows, k, rng, time_budget=min(40.0, _time_left() - 10.0))
        if cdvr_max < best_max:
            best_max = cdvr_max
            best_rows = cdvr_rows

    # Phase 4: Another greedy + descent with different seed if time allows
    if _time_left() > 25.0:
        rng2 = random.Random(123)
        alt_rows, alt_max = _try_greedy_variants(n_rows, k, rng2, num_trials=15)
        if alt_rows is not None:
            alt_refined_rows, alt_refined_max = _targeted_descent(alt_rows, n_rows, k, rng2, time_budget=min(15.0, _time_left() - 10.0))
            if alt_refined_max < best_max:
                best_max = alt_refined_max
                best_rows = alt_refined_rows
            if _time_left() > 15.0:
                cdvr2_rows, cdvr2_max = _local_search_cdvr(alt_refined_rows, n_rows, k, rng2, time_budget=min(20.0, _time_left() - 10.0))
                if cdvr2_max < best_max:
                    best_max = cdvr2_max
                    best_rows = cdvr2_rows

    # Final verification
    if not _validate_solution(best_rows, n_rows, k):
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)

    return best_rows
