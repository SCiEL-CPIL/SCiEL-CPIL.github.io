"""Rewritten program: difference-slot coloring with min-conflict + simulated annealing."""
import time
import random
import math


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Number of difference slots per row
    slots_per_row = k * (k + 1) // 2
    total_slots = n_rows * slots_per_row

    # ---------- helpers ----------
    def pure_greedy():
        """Fast valid baseline: greedy construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diffs(row):
        """Return list of differences for all pairs (i,j) with i<j, in order."""
        out = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                out.append(row[j] - row[i])
        return out

    def diffs_to_row(diffs, k):
        """Given a consistent set of k(k+1)/2 differences (in the standard
        order: for i in range(k+1): for j in range(i+1,k+1): diffs[i*(2k+1-i)//2 + (j-i-1)]
        or simply the order matching row_diffs), reconstruct the row entries.
        Returns None if inconsistent.
        """
        # diffs is in the order: pair (0,1),(0,2),...,(0,k),(1,2),(1,3),...,(k-1,k)
        # We need to recover entries a[0..k] with a[0]=0.
        # a[j] - a[i] = diffs[index]
        # The index for pair (i,j) where i<j:
        # count of pairs (i',j') with i'<i: i*(2k+1-i)//2 ... let me just use a lookup.
        n = k + 1
        # Build index mapping
        idx = 0
        index_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                index_map[(i, j)] = idx
                idx += 1
        # Recover entries: a[0]=0, a[j] = a[j-1] + (a[j]-a[j-1]) but we don't have that directly.
        # Actually a[j] - a[0] = diffs[index_map[(0,j)]]
        # So a[j] = diffs[index_map[(0,j)]] for j>=1, a[0]=0.
        entries = [0] * n
        for j in range(1, n):
            entries[j] = diffs[index_map[(0, j)]]
        # Verify consistency: for all i<j, entries[j]-entries[i] should equal diffs[index_map[(i,j)]]
        for i in range(n):
            for j in range(i + 1, n):
                if entries[j] - entries[i] != diffs[index_map[(i, j)]]:
                    return None
        # Verify strictly increasing
        for j in range(1, n):
            if entries[j] <= entries[j - 1]:
                return None
        return entries

    def build_slot_assignments(rows):
        """Convert rows to slot assignments: list of lists, one per row, each
        containing the difference values in the standard pair order."""
        assignments = []
        for row in rows:
            assignments.append(row_diffs(row))
        return assignments

    def assignments_to_rows(assignments, k):
        """Convert slot assignments back to rows. Returns None if any row is inconsistent."""
        rows = []
        for diffs in assignments:
            row = diffs_to_row(diffs, k)
            if row is None:
                return None
            rows.append(row)
        return rows

    def count_conflicts(assignments):
        """Count the number of duplicate difference values across all slots.
        A conflict is a value that appears more than once."""
        from collections import Counter
        counter = Counter()
        for diffs in assignments:
            for d in diffs:
                if d > 0:
                    counter[d] += 1
        conflicts = sum(1 for v in counter.values() if v > 1)
        # Also count how many slots are involved in conflicts
        involved = sum(v - 1 for v in counter.values() if v > 1)
        return involved  # number of "extra" occurrences

    def scope_from_assignments(assignments):
        """Compute the scope (max entry) from slot assignments."""
        max_scope = 0
        for diffs in assignments:
            row = diffs_to_row(diffs, k)
            if row is not None:
                max_scope = max(max_scope, max(row))
        return max_scope

    # ---------- min-conflict + simulated annealing ----------
    def min_conflict_sa(assignments, time_limit):
        """Min-conflict local search with simulated annealing on slot assignments.
        State: assignments (list of lists of difference values).
        Move: pick a random slot, assign it a new random difference value.
        Cost: number of conflict-involved slots + lambda * scope.
        """
        current = [d[:] for d in assignments]
        current_conflicts = count_conflicts(current)
        current_scope = scope_from_assignments(current)
        current_cost = current_conflicts * 1000 + current_scope

        best = [d[:] for d in current]
        best_conflicts = current_conflicts
        best_scope = current_scope
        best_cost = current_cost

        # Temperature schedule
        T0 = 50.0
        T_end = 0.01
        total_iters = 50000
        # We'll do adaptive iterations based on time

        it = 0
        # Precompute index maps for efficiency
        n = k + 1
        index_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                index_map[(i, j)] = idx
                idx += 1

        # For fast conflict checking, maintain a counter of difference values
        from collections import Counter
        val_counter = Counter()
        for diffs in current:
            for d in diffs:
                if d > 0:
                    val_counter[d] += 1
        conflict_count = sum(1 for v in val_counter.values() if v > 1)
        involved = sum(v - 1 for v in val_counter.values() if v > 1)

        # We'll track per-slot conflict status
        # A slot is in conflict if its value appears >1 times

        while time.time() < time_limit:
            it += 1
            # Cooling
            T = T0 * (T_end / T0) ** (min(it, total_iters) / total_iters)

            # Pick a random slot
            ri = random.randrange(n_rows)
            si = random.randrange(slots_per_row)

            old_val = current[ri][si]

            # Choose new value: random in a range
            # The new value should be a plausible difference
            max_val = max(current_scope, best_scope) + 20
            new_val = random.randint(1, max_val)

            if new_val == old_val:
                continue

            # Apply move
            current[ri][si] = new_val
            val_counter[old_val] -= 1
            if val_counter[old_val] == 0:
                del val_counter[old_val]
            val_counter[new_val] += 1

            # Recompute conflict count incrementally is complex; just recompute
            # For efficiency, track delta
            old_involved = involved
            # Recompute involved
            involved = sum(v - 1 for v in val_counter.values() if v > 1)
            delta_conflicts = involved - old_involved

            # Check if row is still consistent
            row = diffs_to_row(current[ri], k)
            if row is None:
                # Revert
                current[ri][si] = old_val
                val_counter[old_val] -= 1
                if val_counter[old_val] == 0:
                    del val_counter[old_val]
                val_counter[new_val] -= 1
                if val_counter[new_val] == 0:
                    del val_counter[new_val]
                continue

            new_scope = max(max(row), scope_from_assignments([current[i] for i in range(n_rows) if i != ri]))
            # Actually scope_from_assignments recomputes all; let's track max per row
            # For now, just compute
            current_cost = involved * 1000 + new_scope
            best_cost_check = best_conflicts * 1000 + best_scope

            if current_cost < best_cost_check:
                best = [d[:] for d in current]
                best_conflicts = involved
                best_scope = new_scope
                best_cost = current_cost

            # Accept/reject
            if delta_conflicts < 0 or current_cost < best_cost_check:
                pass  # accept (improvement)
            else:
                delta_cost = current_cost - (old_involved * 1000 + current_scope)
                # Actually we need the old cost
                # Let me restructure: compute old_cost before move
                # For now, use simple: accept if not worse, or with probability
                if delta_conflicts > 0:
                    prob = math.exp(-delta_conflicts * 1000 / max(T, 0.001))
                    if random.random() > prob:
                        # Revert
                        current[ri][si] = old_val
                        val_counter[old_val] -= 1
                        if val_counter[old_val] == 0:
                            del val_counter[old_val]
                        val_counter[new_val] -= 1
                        if val_counter[new_val] == 0:
                            del val_counter[new_val]
                        involved = old_involved
                        continue

        return best, best_scope

    # ---------- main search ----------
    # Baseline
    base_rows = pure_greedy()
    best_rows = base_rows
    best_scope = scope_of(base_rows)

    # Convert to slot assignments
    base_assignments = build_slot_assignments(base_rows)

    # Run min-conflict SA
    improved_assignments, improved_scope = min_conflict_sa(base_assignments, deadline)
    if improved_scope < best_scope:
        improved_rows = assignments_to_rows(improved_assignments, k)
        if improved_rows is not None:
            best_rows = improved_rows
            best_scope = improved_scope

    # If still above counting bound, try a few restarts with different initializations
    if best_scope > 112 and time.time() < deadline - 10:
        for restart in range(3):
            if time.time() >= deadline - 5:
                break
            # Randomized greedy for diverse start
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    # Randomize: pick from a window
                    window = 40
                    cands = list(range(x + 1, x + window + 1))
                    random.shuffle(cands)
                    found = False
                    for cand in cands:
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            found = True
                            break
                    if not found:
                        cand = x + 1
                        while True:
                            new = {cand - a for a in row}
                            if not new & used:
                                row.append(cand)
                                used |= new
                                x = cand
                                break
                            cand += 1
                rows.append(row)
            sc = scope_of(rows)
            if sc < best_scope:
                best_rows = rows
                best_scope = sc
            # Run SA on this start
            assignments = build_slot_assignments(rows)
            improved_a, improved_sc = min_conflict_sa(assignments, deadline)
            if improved_sc < best_scope:
                improved_rows = assignments_to_rows(improved_a, k)
                if improved_rows is not None:
                    best_rows = improved_rows
                    best_scope = improved_sc
            if best_scope <= 112:
                break

    return best_rows
