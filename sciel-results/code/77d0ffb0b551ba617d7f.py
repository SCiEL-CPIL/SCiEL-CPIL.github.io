"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Per-row local optimization (refinement machinery) ----------
    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def try_reduce_row(rows, row_idx, used_other, max_attempts=200):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 200):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        while improved and time_left() > 12:
            improved = False
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 12:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=250)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(best_rows)

    # ---------- Simulated Annealing (exploration machinery) ----------
    # Conflict counting via integer array for fast delta evaluation.
    if time_left() > 40:
        n_diffs_per_row = k * (k + 1) // 2  # number of differences per row
        # Order rows by descending scope: index 0 is the binding constraint.
        order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
        sa_rows = [list(best_rows[i]) for i in order]

        # Global conflict count array.
        max_diff = max(max(r) for r in sa_rows) + 1
        conflict = [0] * (max_diff + 1)
        for row in sa_rows:
            for d in row_diffs_list(row):
                if d > max_diff:
                    max_diff = d
                    conflict = [0] * (max_diff + 1)
                conflict[d] += 1

        # Track number of conflicts incrementally
        cur_conflict_count = sum(1 for c in conflict if c > 1)

        def total_conflicts():
            nonlocal cur_conflict_count
            return cur_conflict_count

        def update_conflict(d, delta):
            """Update conflict count when adding/removing a difference d."""
            nonlocal cur_conflict_count
            old = conflict[d]
            new = old + delta
            if old == 1 and new == 2:
                cur_conflict_count += 1
            elif old == 2 and new == 1:
                cur_conflict_count -= 1
            conflict[d] = new

        def row_diffs_counts(row):
            """Return (diffs list, internal duplicate count) for a row."""
            counts = {}
            dup = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in counts:
                        dup += 1
                    counts[d] = counts.get(d, 0) + 1
            return list(counts.keys()), dup

        def remove_row_diffs(row):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    update_conflict(d, -1)

        def add_row_diffs(row):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d >= len(conflict):
                        conflict.extend([0] * (d + 1 - len(conflict)))
                    update_conflict(d, 1)

        def energy():
            # Primary: total conflicts. Secondary: scope of the largest row (index 0).
            return (total_conflicts(), max(sa_rows[0]))

        cur_conf, cur_scope = energy()
        best_sa = [list(r) for r in sa_rows]
        best_energy = (cur_conf, cur_scope)

        # Precompute per-row differences for fast delta.
        row_diffs_cache = [row_diffs_counts(r)[0] for r in sa_rows]

        def greedy_row_for(used_set):
            """Greedy row given a set of already-used differences (from other rows)."""
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
            return row

        def random_row_for(used_set, jitter=0.05):
            """Greedy row with occasional skips to explore alternatives."""
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    if jitter and random.random() < jitter and x < 300:
                        continue
                    row.append(x)
                    row_diffs |= new
            return row

        # Precompute per-row difference sets for fast used-set construction
        row_diffs_sets = [set(row_diffs_cache[i]) for i in range(n_rows)]

        # SA loop: pick a row, remove it, propose a new row, evaluate delta.
        t = 1.0
        t_min = 0.01
        cooling = 0.997
        step = 0
        while time_left() > 20 and t > t_min:
            step += 1
            # Pick a row: bias toward the largest (index 0) early, random later.
            if random.random() < 0.4:
                idx = 0
            else:
                idx = random.randrange(n_rows)

            old_row = sa_rows[idx]
            old_diffs = row_diffs_cache[idx]

            # Remove old row's differences.
            remove_row_diffs(old_row)
            conf_after_remove = total_conflicts()

            # Build used set from remaining rows (use precomputed sets)
            used_set = set()
            for j in range(n_rows):
                if j != idx:
                    used_set.update(row_diffs_sets[j])

            # Propose new row with more exploration.
            r = random.random()
            if r < 0.5:
                new_row = greedy_row_for(used_set)
            elif r < 0.8:
                new_row = random_row_for(used_set, jitter=0.15)
            else:
                # More aggressive exploration: random restart of row
                new_row = random_row_for(used_set, jitter=0.3)

            # Ensure new row has exactly k+1 entries (greedy always does).
            if len(new_row) != k + 1:
                # Fallback: revert.
                add_row_diffs(old_row)
                t *= cooling
                continue

            # Add new row's differences, compute new conflict count.
            new_diffs = row_diffs_counts(new_row)[0]
            new_diffs_set = set(new_diffs)
            add_row_diffs(new_row)
            conf_after_add = total_conflicts()

            new_scope = max(new_row) if idx == 0 else max(sa_rows[0])
            # Energy delta: (conflict change, scope change)
            conf_delta = conf_after_add - conf_after_remove
            scope_delta = new_scope - cur_scope

            accept = False
            if conf_delta < 0 or (conf_delta == 0 and scope_delta < 0):
                accept = True
            else:
                # Metropolis on combined penalty.
                penalty = conf_delta * 1000 + scope_delta
                if penalty < 0:
                    accept = True
                else:
                    if random.random() < pow(2.718281828, -penalty / max(t, 1e-6)):
                        accept = True

            if accept:
                sa_rows[idx] = new_row
                row_diffs_cache[idx] = new_diffs
                row_diffs_sets[idx] = new_diffs_set
                cur_conf = conf_after_add
                cur_scope = new_scope
                cur_energy = (cur_conf, cur_scope)
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_sa = [list(r) for r in sa_rows]
            else:
                # Revert.
                remove_row_diffs(new_row)
                add_row_diffs(old_row)

            t *= cooling

        # Reorder best_sa back to original row order.
        if best_energy[0] == 0:  # valid
            # Map back: best_sa[i] corresponds to original row order[i].
            restored = [None] * n_rows
            for i in range(n_rows):
                restored[order[i]] = best_sa[i]
            if is_valid(restored):
                s = scope(restored)
                if s < best_scope:
                    best_rows = restored
                    best_scope = s

        # Final local polish after SA.
        if time_left() > 15:
            polished = local_polish(best_rows)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)

        # Multi-restart: if time remains, try again from a different random seed
        if time_left() > 40:
            random.seed(random.randint(0, 1000000))
            # Re-run SA with fresh randomization
            sa_rows2 = [list(best_rows[i]) for i in order]
            max_diff2 = max(max(r) for r in sa_rows2) + 1
            conflict2 = [0] * (max_diff2 + 1)
            for row in sa_rows2:
                for d in row_diffs_list(row):
                    if d > max_diff2:
                        max_diff2 = d
                        conflict2 = [0] * (max_diff2 + 1)
                    conflict2[d] += 1

            cur_conflict_count2 = sum(1 for c in conflict2 if c > 1)

            def total_conflicts2():
                nonlocal cur_conflict_count2
                return cur_conflict_count2

            def update_conflict2(d, delta):
                nonlocal cur_conflict_count2
                old = conflict2[d]
                new = old + delta
                if old == 1 and new == 2:
                    cur_conflict_count2 += 1
                elif old == 2 and new == 1:
                    cur_conflict_count2 -= 1
                conflict2[d] = new

            def remove_row_diffs2(row):
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        update_conflict2(d, -1)

            def add_row_diffs2(row):
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d >= len(conflict2):
                            conflict2.extend([0] * (d + 1 - len(conflict2)))
                        update_conflict2(d, 1)

            row_diffs_cache2 = [row_diffs_counts(r)[0] for r in sa_rows2]
            row_diffs_sets2 = [set(row_diffs_cache2[i]) for i in range(n_rows)]

            cur_conf2, cur_scope2 = (cur_conflict_count2, max(sa_rows2[0]))
            best_sa2 = [list(r) for r in sa_rows2]
            best_energy2 = (cur_conf2, cur_scope2)

            t2 = 1.0
            t_min2 = 0.01
            cooling2 = 0.997
            while time_left() > 20 and t2 > t_min2:
                idx = 0 if random.random() < 0.4 else random.randrange(n_rows)
                old_row2 = sa_rows2[idx]
                remove_row_diffs2(old_row2)
                conf_after_remove2 = total_conflicts2()
                used_set2 = set()
                for j in range(n_rows):
                    if j != idx:
                        used_set2.update(row_diffs_sets2[j])
                r2 = random.random()
                if r2 < 0.5:
                    new_row2 = greedy_row_for(used_set2)
                elif r2 < 0.8:
                    new_row2 = random_row_for(used_set2, jitter=0.15)
                else:
                    new_row2 = random_row_for(used_set2, jitter=0.3)
                if len(new_row2) != k + 1:
                    add_row_diffs2(old_row2)
                    t2 *= cooling2
                    continue
                new_diffs2 = row_diffs_counts(new_row2)[0]
                new_diffs_set2 = set(new_diffs2)
                add_row_diffs2(new_row2)
                conf_after_add2 = total_conflicts2()
                new_scope2 = max(new_row2) if idx == 0 else max(sa_rows2[0])
                conf_delta2 = conf_after_add2 - conf_after_remove2
                scope_delta2 = new_scope2 - cur_scope2
                accept2 = False
                if conf_delta2 < 0 or (conf_delta2 == 0 and scope_delta2 < 0):
                    accept2 = True
                else:
                    penalty2 = conf_delta2 * 1000 + scope_delta2
                    if penalty2 < 0:
                        accept2 = True
                    else:
                        if random.random() < pow(2.718281828, -penalty2 / max(t2, 1e-6)):
                            accept2 = True
                if accept2:
                    sa_rows2[idx] = new_row2
                    row_diffs_cache2[idx] = new_diffs2
                    row_diffs_sets2[idx] = new_diffs_set2
                    cur_conf2 = conf_after_add2
                    cur_scope2 = new_scope2
                    cur_energy2 = (cur_conf2, cur_scope2)
                    if cur_energy2 < best_energy2:
                        best_energy2 = cur_energy2
                        best_sa2 = [list(r) for r in sa_rows2]
                else:
                    remove_row_diffs2(new_row2)
                    add_row_diffs2(old_row2)
                t2 *= cooling2

            if best_energy2[0] == 0:
                restored2 = [None] * n_rows
                for i in range(n_rows):
                    restored2[order[i]] = best_sa2[i]
                if is_valid(restored2):
                    s2 = scope(restored2)
                    if s2 < best_scope:
                        best_rows = restored2
                        best_scope = s2

        # Final polish after all SA
        if time_left() > 10:
            polished = local_polish(best_rows)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
