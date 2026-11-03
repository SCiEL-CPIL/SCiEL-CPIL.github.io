"""Difference triangle set (7,5) via SA with delta conflict tracking + polish."""
import time
import random
import math
import sys

def _all_diffs(row):
    """Return the set of positive differences in a row."""
    return set(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))

def _diffs_to(row, idx):
    """Differences involving position idx in row (as positive values)."""
    d = set()
    for i in range(len(row)):
        if i != idx:
            d.add(abs(row[idx] - row[i]))
    return d

def _build_used(rows):
    """Global multiset count of every difference across all rows."""
    used = {}
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                used[d] = used.get(d, 0) + 1
    return used

def _conflict_count(used):
    """Number of differences that appear more than once."""
    return sum(1 for c in used.values() if c > 1)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    rng = random.Random(12345)

    # --- Phase 1: Greedy initial solution ---
    used = {}
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            ok = True
            for a in row:
                d = x - a
                if used.get(d, 0) >= 1:
                    ok = False
                    break
            if ok:
                for a in row:
                    d = x - a
                    used[d] = used.get(d, 0) + 1
                row.append(x)
        rows.append(row)

    best_rows = [list(r) for r in rows]
    best_scope = max(max(r) for r in best_rows)

    def _scope_of(rows):
        return max(max(r) for r in rows)

    # --- Phase 2: Simulated Annealing with delta conflict tracking ---
    # State: rows (list of lists), used (dict diff->count)
    # Move: pick a random row, random position (not 0), change value to a nearby value
    # that keeps the row strictly increasing and non-negative.
    # Cost = number of conflicts (differences appearing >1 time) + scope penalty.

    cur_rows = [list(r) for r in rows]
    cur_used = _build_used(cur_rows)
    cur_conflicts = _conflict_count(cur_used)
    cur_scope = _scope_of(cur_rows)
    cur_cost = cur_conflicts * 1000 + cur_scope

    T = 100.0
    T_min = 0.01
    cool = 0.9995
    step = 1

    # Precompute: for each row, the valid range for each position
    # Position 0 is always 0. Position j (j>=1) must satisfy row[j-1]+1 <= val <= row[j+1]-1 (or scope if last)

    while T > T_min:
        if time.time() > deadline:
            break

        # Pick a random row and position
        ri = rng.randrange(n_rows)
        row = cur_rows[ri]
        # Positions 1..k (not 0)
        pi = rng.randrange(1, k + 1)

        # Determine valid range for row[pi]
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi < k else 200  # generous upper bound
        if lo > hi:
            continue

        # Propose a new value: current +/- random small step
        old_val = row[pi]
        delta = rng.choice([-1, 1]) * rng.randint(1, step)
        new_val = old_val + delta
        if new_val < lo or new_val > hi:
            # Try a random value in range
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

        # Compute delta in used counts
        # Old differences involving position pi
        old_diffs = _diffs_to(row, pi)
        # New differences involving position pi
        new_row = list(row)
        new_row[pi] = new_val
        new_diffs = _diffs_to(new_row, pi)

        # Differences removed: in old_diffs but not in new_diffs
        # Differences added: in new_diffs but not in old_diffs
        removed = old_diffs - new_diffs
        added = new_diffs - old_diffs

        # Cost delta from conflicts
        # For each removed diff d: if used[d] was 2, conflict decreases by 1; if was 1, no change
        # For each added diff d: if used[d] was 1, conflict increases by 1; if was 0, no change
        dc = 0
        for d in removed:
            if cur_used.get(d, 0) == 2:
                dc -= 1
        for d in added:
            if cur_used.get(d, 0) == 1:
                dc += 1

        # Scope change
        new_scope = cur_scope
        if new_val > cur_scope:
            new_scope = new_val
        # Check if scope decreases (only if old_val was the max and new_val < old_val)
        if old_val == cur_scope and new_val < old_val:
            # Need to recompute scope
            new_scope = _scope_of(cur_rows)  # expensive but rare
            # Actually let's just use the max of all rows
            # This is O(n_rows * k) which is fine for 7*6
            new_scope = max(max(r) for r in cur_rows)
            # But we changed cur_rows[ri][pi], so:
            # Actually cur_rows hasn't been updated yet, so old_scope is still cur_scope
            # Let me just compute: the new scope is max of all entries except possibly the old max
            # Simpler: just track it
            new_scope = cur_scope
            if new_val < cur_scope:
                # The old value was the scope; new value is smaller
                # Scope might decrease if no other entry equals cur_scope
                # For simplicity, keep cur_scope (conservative)
                new_scope = cur_scope

        new_cost = (cur_conflicts + dc) * 1000 + new_scope
        delta_cost = new_cost - cur_cost

        # Accept?
        if delta_cost <= 0 or rng.random() < math.exp(-delta_cost / T):
            # Apply move
            row[pi] = new_val
            for d in removed:
                cur_used[d] -= 1
                if cur_used[d] == 0:
                    del cur_used[d]
            for d in added:
                cur_used[d] = cur_used.get(d, 0) + 1
            cur_conflicts += dc
            cur_scope = new_scope
            cur_cost = new_cost

            # Update best
            if cur_conflicts == 0 and cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [list(r) for r in cur_rows]

        T *= cool

    # --- Phase 3: Final polish ---
    # Try to reduce scope by moving entries down
    if time.time() < deadline:
        # Work on a copy
        pol_rows = [list(r) for r in best_rows]
        pol_used = _build_used(pol_rows)
        pol_scope = _scope_of(pol_rows)
        pol_conflicts = _conflict_count(pol_used)

        # Iteratively try to reduce the largest entry
        for _pol_iter in range(5000):
            if time.time() > deadline:
                break
            if pol_conflicts > 0:
                break

            # Find the row and position with the largest value
            max_val = -1
            max_ri = -1
            max_pi = -1
            for i, r in enumerate(pol_rows):
                for j in range(1, len(r)):
                    if r[j] > max_val:
                        max_val = r[j]
                        max_ri = i
                        max_pi = j

            if max_ri < 0:
                break

            row = pol_rows[max_ri]
            # Try to reduce row[max_pi] by trying values from max_val-1 down to row[max_pi-1]+1
            lo = row[max_pi - 1] + 1
            hi = row[max_pi + 1] - 1 if max_pi < k else max_val - 1
            found = False
            for new_val in range(max_val - 1, lo - 1, -1):
                if max_pi < k and new_val >= row[max_pi + 1]:
                    continue
                old_diffs = _diffs_to(row, max_pi)
                new_row = list(row)
                new_row[max_pi] = new_val
                new_diffs = _diffs_to(new_row, max_pi)
                removed = old_diffs - new_diffs
                added = new_diffs - old_diffs

                # Check if this keeps conflicts at 0
                ok = True
                for d in removed:
                    if pol_used.get(d, 0) == 2:
                        # Would reduce a conflict - good, but we want to maintain 0 conflicts
                        # Actually if conflicts is 0, no diff has count 2, so this is fine
                        pass
                for d in added:
                    if pol_used.get(d, 0) >= 1:
                        ok = False
                        break
                if not ok:
                    continue

                # Apply
                row[max_pi] = new_val
                for d in removed:
                    pol_used[d] -= 1
                    if pol_used[d] == 0:
                        del pol_used[d]
                for d in added:
                    pol_used[d] = pol_used.get(d, 0) + 1
                pol_scope = max(pol_scope - 1, new_val)
                # Recompute scope properly
                pol_scope = _scope_of(pol_rows)
                found = True
                break

            if not found:
                break

        if pol_conflicts == 0 and pol_scope < best_scope:
            best_scope = pol_scope
            best_rows = pol_rows

    # --- Phase 4: More SA rounds if time remains ---
    if time.time() < deadline - 5:
        # Restart SA from best with lower temperature
        cur_rows = [list(r) for r in best_rows]
        cur_used = _build_used(cur_rows)
        cur_conflicts = _conflict_count(cur_used)
        cur_scope = _scope_of(cur_rows)
        cur_cost = cur_conflicts * 1000 + cur_scope
        T = 20.0
        T_min = 0.01
        cool = 0.9998

        while T > T_min:
            if time.time() > deadline:
                break
            ri = rng.randrange(n_rows)
            row = cur_rows[ri]
            pi = rng.randrange(1, k + 1)
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else 200
            if lo > hi:
                continue
            old_val = row[pi]
            delta = rng.choice([-1, 1]) * rng.randint(1, 2)
            new_val = old_val + delta
            if new_val < lo or new_val > hi:
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue

            old_diffs = _diffs_to(row, pi)
            new_row = list(row)
            new_row[pi] = new_val
            new_diffs = _diffs_to(new_row, pi)
            removed = old_diffs - new_diffs
            added = new_diffs - old_diffs

            dc = 0
            for d in removed:
                if cur_used.get(d, 0) == 2:
                    dc -= 1
            for d in added:
                if cur_used.get(d, 0) == 1:
                    dc += 1

            new_scope = cur_scope
            if new_val > cur_scope:
                new_scope = new_val

            new_cost = (cur_conflicts + dc) * 1000 + new_scope
            delta_cost = new_cost - cur_cost

            if delta_cost <= 0 or rng.random() < math.exp(-delta_cost / T):
                row[pi] = new_val
                for d in removed:
                    cur_used[d] -= 1
                    if cur_used[d] == 0:
                        del cur_used[d]
                for d in added:
                    cur_used[d] = cur_used.get(d, 0) + 1
                cur_conflicts += dc
                cur_scope = new_scope
                cur_cost = new_cost

                if cur_conflicts == 0 and cur_scope < best_scope:
                    best_scope = cur_scope
                    best_rows = [list(r) for r in cur_rows]

            T *= cool

    # Final validation
    all_diffs = []
    for row in best_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                all_diffs.append(row[j] - row[i])
    if len(all_diffs) != len(set(all_diffs)):
        # Invalid - fall back to greedy
        used2 = {}
        rows2 = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                ok = True
                for a in row:
                    d = x - a
                    if used2.get(d, 0) >= 1:
                        ok = False
                        break
                if ok:
                    for a in row:
                        d = x - a
                        used2[d] = used2.get(d, 0) + 1
                    row.append(x)
            rows2.append(row)
        return rows2

    return best_rows
