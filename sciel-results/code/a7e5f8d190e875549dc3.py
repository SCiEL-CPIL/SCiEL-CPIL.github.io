import time
import random
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 132.0  # safety margin before 150s kill

    d = k * (k + 1) // 2  # differences per row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def greedy_init():
        """Greedy construction with smarter skip: try multiple candidates per step."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                for trial in range(x + 1, x + 50):
                    new = {trial - a for a in row}
                    if not new & used:
                        row.append(trial)
                        used |= new
                        x = trial
                        found = True
                        break
                if not found:
                    while True:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            break
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(12345)

    # ============================================================
    # Deterministic backtracking with forward checking
    # Model: each row is a sequence of gap variables.
    # Row r has gaps g[0], g[1], ..., g[k] where g[0] >= 1, g[i] >= 1 for i>0
    # Entry j = sum(g[0..j-1]). Scope = sum of all gaps in the largest row.
    # Constraint: all pairwise differences across all rows must be distinct.
    # ============================================================

    def backtracking_solve(initial_rows, max_scope, time_budget):
        """
        Deterministic backtracking with forward checking.
        Variables: gaps for each row, ordered by tightest capacity first.
        Forward checking: maintain set of used differences; prune if adding
        new differences would conflict.
        Returns (rows, scope) if found within time_budget, else (None, None).
        """
        start_time = time.time()
        n = k + 1  # entries per row

        # Represent each row by its gaps
        # gaps[r] = list of k+1 positive integers
        # entry[j] = sum(gaps[r][0..j-1]) for j=1..k, entry[0]=0
        # scope of row r = sum(gaps[r])

        # Initial gaps from initial_rows
        init_gaps = []
        for r in initial_rows:
            gaps = []
            for j in range(1, n):
                gaps.append(r[j] - r[j-1])
            init_gaps.append(gaps)

        # Total number of differences needed
        total_diffs = n_rows * d

        # Forward checking state
        used_diffs = set()
        # For each row, track which diffs are "locked in" by its gaps
        row_locked_diffs = [set() for _ in range(n_rows)]

        # Compute diffs for a row given its gaps
        def compute_row_diffs(gaps):
            """Given gaps list, return set of all pairwise differences."""
            entries = [0]
            for g in gaps:
                entries.append(entries[-1] + g)
            diffs = set()
            for i in range(n):
                for j in range(i+1, n):
                    diffs.add(entries[j] - entries[i])
            return diffs, entries

        # Check if gaps for row r can be extended without conflict
        def can_extend(row_idx, new_gaps, used_by_others):
            """Check if setting row row_idx to new_gaps creates no conflicts."""
            diffs, entries = compute_row_diffs(new_gaps)
            if len(diffs) < d:
                return False  # internal duplicates
            if diffs & used_by_others:
                return False
            return True, diffs, entries

        # Dynamic variable ordering: pick row with fewest valid gap assignments
        # For now, use a heuristic: process rows in order of their current max entry
        # (tightest first)

        # Backtracking search
        # We'll search over gap assignments row by row
        # For each row, we try to find gaps that fit within remaining scope budget

        best_found = None
        best_found_scope = max_scope + 1

        # Pre-compute: for each row index, what's the minimum possible scope
        # given the diffs already used by other rows
        def min_possible_scope(row_idx, used_by_others, current_scope_budget):
            """Lower bound on scope for row_idx given constraints."""
            # Simple lower bound: need d distinct positive diffs, all <= scope
            # Minimum scope to have d distinct diffs from {1,...,scope} is when
            # we use diffs 1,2,...,d, but that requires specific gap structure.
            # Conservative bound: scope >= d (since we need d distinct positive ints)
            return d

        def backtrack(row_idx, gaps_so_far, used_diffs_so_far, scope_so_far, remaining_budget):
            """
            Recursively assign gaps for rows row_idx..n_rows-1.
            gaps_so_far: list of gap lists for rows 0..row_idx-1
            used_diffs_so_far: set of all diffs used so far
            scope_so_far: max scope among assigned rows
            remaining_budget: max_scope - scope_so_far (remaining room)
            """
            nonlocal best_found, best_found_scope

            if time.time() - start_time > time_budget:
                return False

            if row_idx == n_rows:
                scope = scope_so_far
                if scope < best_found_scope:
                    best_found_scope = scope
                    best_found = [None] * n_rows
                    for ri in range(n_rows):
                        entries = [0]
                        for g in gaps_so_far[ri]:
                            entries.append(entries[-1] + g)
                        best_found[ri] = entries
                return True

            # Dynamic ordering: among unassigned rows, pick the one with tightest constraint
            # For simplicity, process in order but with a twist: try to assign
            # the row that has the most constrained gap space first.
            # Heuristic: rows with larger current scope are more constrained.
            # For now, use fixed order (row_idx) but with forward checking.

            # Forward checking: compute used diffs from other already-assigned rows
            used_by_others = set(used_diffs_so_far)

            # Try gap assignments for this row
            # The row needs k+1 gaps, each >= 1, sum <= remaining_budget
            # We search for gaps that produce d distinct diffs, none in used_by_others

            # Strategy: use a sub-backtracking for the gaps of this row
            # Gaps are ordered: g[0], g[1], ..., g[k]
            # Entry j = sum(g[0..j-1])
            # We build entries incrementally

            def assign_gaps(gap_idx, current_entry, partial_gaps, partial_diffs, partial_entries):
                nonlocal best_found, best_found_scope

                if time.time() - start_time > time_budget:
                    return False

                if gap_idx == k + 1:
                    # All gaps assigned for this row
                    scope_this_row = current_entry
                    new_scope = max(scope_so_far, scope_this_row)
                    if new_scope > max_scope:
                        return False

                    # Check all diffs are distinct from each other and from used_by_others
                    if len(partial_diffs) != d:
                        return False
                    if partial_diffs & used_by_others:
                        return False

                    # Forward check: can remaining rows still be assigned?
                    new_used = used_by_others | partial_diffs
                    remaining_rows = n_rows - row_idx - 1
                    # Simple check: remaining rows need d diffs each, all distinct
                    # and not in new_used. Total available diffs up to max_scope is max_scope.
                    # Need len(new_used) + remaining_rows * d <= max_scope
                    if len(new_used) + remaining_rows * d > max_scope:
                        return False

                    # Recurse
                    gaps_so_far[row_idx] = list(partial_gaps)
                    if backtrack(row_idx + 1, gaps_so_far, new_used, new_scope, max_scope - new_scope):
                        return True

                    return False

                # Determine range for gap[gap_idx]
                # current_entry is the entry at position gap_idx (0-indexed entries)
                # Next entry = current_entry + gap[gap_idx]
                # Max next entry is bounded by max_scope
                max_next_entry = max_scope
                min_gap = 1
                max_gap = max_next_entry - current_entry

                if max_gap < min_gap:
                    return False

                # Additional constraint: partial diffs must remain feasible
                # The new diffs added by setting this gap:
                # For each previous entry e in partial_entries, diff = (current_entry + gap) - e
                # These must not be in partial_diffs or used_by_others

                # Try gaps in increasing order (to find small scopes first)
                for gap_val in range(min_gap, max_gap + 1):
                    if time.time() - start_time > time_budget:
                        return False

                    next_entry = current_entry + gap_val

                    # Compute new diffs
                    new_diffs = set()
                    valid = True
                    for prev_e in partial_entries:
                        diff = next_entry - prev_e
                        if diff in new_diffs:
                            valid = False
                            break
                        if diff in partial_diffs:
                            valid = False
                            break
                        if diff in used_by_others:
                            valid = False
                            break
                        new_diffs.add(diff)

                    if not valid:
                        continue

                    # Also check: the new entry doesn't create issues
                    # (already checked via diffs)

                    partial_gaps.append(gap_val)
                    partial_entries.append(next_entry)
                    partial_diffs |= new_diffs

                    if assign_gaps(gap_idx + 1, next_entry, partial_gaps, partial_diffs, partial_entries):
                        return True

                    partial_gaps.pop()
                    partial_entries.pop()
                    partial_diffs -= new_diffs

                return False

            # Start assigning gaps for this row
            # Initial entry is 0
            initial_entries = [0]
            initial_diffs = set()
            initial_gaps = []

            # Try different max_scope values? No, we're searching within max_scope.
            # But we can try to minimize: search gaps in order that minimizes final entry.

            result = assign_gaps(0, 0, initial_gaps, initial_diffs, initial_entries)
            return result

        # Main search: try decreasing max_scope from current best down
        # Start from best_scope - 1 and go down
        for target_scope in range(best_scope - 1, d - 1, -1):
            if time.time() - start_time > time_budget:
                break

            # Reset
            best_found = None
            best_found_scope = target_scope + 1

            # Initialize gaps_so_far
            gaps_so_far = [None] * n_rows
            used_diffs_so_far = set()

            # Try to find a solution with max scope = target_scope
            if backtrack(0, gaps_so_far, used_diffs_so_far, 0, target_scope):
                if best_found is not None:
                    return best_found, best_found_scope

        return None, None

    # ============================================================
    # Polish functions (kept from before)
    # ============================================================

    def rebuild_row(used_other, jitter=0.0, max_val=None, local_rng=None):
        if local_rng is None:
            local_rng = rng
        row = [0]
        row_diffs = set()
        x = 0
        search_end = max_val if max_val is not None else (best_scope + 30)
        while len(row) < k + 1:
            found = False
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and local_rng.random() < jitter and trial_x < 300:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def local_polish(rows, max_passes=8):
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 12 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 12:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                best_row = list(rows[i])
                best_max = max(rows[i])
                for attempt in range(40):
                    if attempt < 15:
                        jitter = 0.0
                    elif attempt < 25:
                        jitter = 0.05 + 0.02 * (attempt - 15)
                    else:
                        jitter = 0.25 + 0.03 * (attempt - 25)
                    new_row = rebuild_row(used_other, jitter=jitter, max_val=best_max - 1)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        return rows

    def targeted_descent(rows, budget=3.0):
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False

        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(15):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    scope = full_validate(cur)
                    if scope is not None and scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break
                continue

            for new_val in range(old_val - 1, lower - 1, -1):
                row[pos] = new_val
                scope = full_validate(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    else:
                        continue
                row[pos] = old_val
                break

            if not improved:
                break

        if improved:
            return cur, cur_scope
        return None, None

    def exact_local_polish(rows, budget=4.0):
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        best_cur = [r[:] for r in cur]
        best_sc = cur_scope
        improved = True

        while improved and time.time() - start < budget:
            improved = False
            for ri in range(n_rows):
                if time.time() - start > budget:
                    break
                row = cur[ri]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() - start > budget:
                        break
                    old_val = row[pos]
                    lo = row[pos - 1] + 1
                    hi = row[pos + 1] - 1 if pos < len(row) - 1 else cur_scope + 10
                    if hi <= lo:
                        continue
                    for new_val in range(old_val - 1, lo - 1, -1):
                        row[pos] = new_val
                        scope = full_validate(cur)
                        if scope is not None:
                            if scope < best_sc:
                                best_sc = scope
                                best_cur = [r[:] for r in cur]
                            improved = True
                            break
                        else:
                            row[pos] = old_val
                            break
                    if improved:
                        break
                if improved:
                    cur = [r[:] for r in best_cur]
                    cur_scope = best_sc

        return best_cur, best_sc

    # ============================================================
    # Phase 1: Deterministic backtracking to find a good initial solution
    # ============================================================
    bt_budget = min(40.0, deadline - time.time() - 30.0)
    if bt_budget > 5.0:
        bt_rows, bt_scope = backtracking_solve(best_rows, best_scope - 1, bt_budget)
        if bt_rows is not None and bt_scope < best_scope:
            best_scope = bt_scope
            best_rows = bt_rows

    # ============================================================
    # Phase 2: Local polish
    # ============================================================
    if time.time() < deadline - 20:
        polished = local_polish(best_rows)
        ps = full_validate(polished)
        if ps is not None and ps < best_scope:
            best_scope = ps
            best_rows = polished

    # ============================================================
    # Phase 3: ILS with kicks
    # ============================================================
    def kick(rows, num_kicks=1):
        rows = [list(r) for r in rows]
        indices = rng.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            success = False
            for attempt in range(20):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row(used_other, jitter=jitter, max_val=best_scope + 30)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, jitter=0.0, max_val=best_scope + 40)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    il_budget = deadline - time.time() - 25.0
    if il_budget > 5.0:
        max_il_iter = 200
        for it in range(max_il_iter):
            if time.time() > deadline - 25:
                break

            cur = [r[:] for r in best_rows]
            if rng.random() < 0.7:
                cur = kick(cur, num_kicks=rng.choice([1, 1, 2]))
            else:
                i = rng.randrange(n_rows)
                j = rng.randrange(n_rows)
                if i != j:
                    cur[i], cur[j] = cur[j], cur[i]

            cur_scope = full_validate(cur)
            if cur_scope is None:
                continue

            if cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [r[:] for r in cur]

            if time.time() < deadline - 20:
                polished = local_polish(cur)
                ps = full_validate(polished)
                if ps is not None and ps < best_scope:
                    best_scope = ps
                    best_rows = [r[:] for r in polished]
                    cur = [r[:] for r in polished]

            if time.time() < deadline - 15:
                desc_rows, desc_scope = targeted_descent(cur, budget=1.0)
                if desc_rows is not None and desc_scope < best_scope:
                    best_scope = desc_scope
                    best_rows = desc_rows
                    cur = [r[:] for r in desc_rows]

    # ============================================================
    # Phase 4: Exact local polish
    # ============================================================
    if time.time() < deadline - 8:
        polished, ps = exact_local_polish(best_rows, budget=5.0)
        if ps < best_scope:
            best_scope = ps
            best_rows = polished

    # ============================================================
    # Phase 5: Final targeted descent
    # ============================================================
    if time.time() < deadline - 3:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=2.0)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    # ============================================================
    # Phase 6: Final exact polish
    # ============================================================
    if time.time() < deadline - 0.5:
        remaining = deadline - time.time() - 0.5
        if remaining > 1.0:
            polished, ps = exact_local_polish(best_rows, budget=remaining)
            if ps < best_scope:
                best_scope = ps
                best_rows = polished

    # Final validation
    final_scope = full_validate(best_rows)
    if final_scope is None:
        # Fallback to greedy
        best_rows = greedy_init()

    return best_rows
