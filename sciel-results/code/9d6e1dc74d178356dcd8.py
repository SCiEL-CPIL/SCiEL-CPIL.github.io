"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    L = k + 1  # length of each row

    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diffs_set(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def randomized_greedy(window=80):
        """Greedy with shuffled candidate window."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
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
        return rows

    def simulated_annealing(rows, time_limit):
        """
        Simulated annealing on the row entries.
        State: list of rows (each strictly increasing, starting at 0).
        Energy: scope (max entry).
        Moves: pick a random row and a random position (1..L-1), try to change
               that entry to a nearby value while preserving strict increase
               and global difference distinctness.
        Accept: always accept if valid and scope doesn't increase; accept worse
               moves with probability exp(-delta/T).
        """
        rows = [r[:] for r in rows]
        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        current_scope = best_scope

        # Precompute diff sets
        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s

        # Temperature schedule
        T0 = 10.0
        T_min = 0.1
        # We'll do a fixed number of iterations per "restart" and reheat if stuck
        total_iters = 0
        stuck_count = 0
        max_stuck = 20000  # reheat after this many non-improving iterations

        while time.time() < time_limit:
            # Compute temperature based on progress
            frac = min(1.0, total_iters / 200000.0)
            T = T0 * (T_min / T0) ** frac
            if T < T_min:
                T = T_min

            total_iters += 1

            # Pick random row and position
            ri = random.randrange(n_rows)
            ci = random.randrange(1, L)

            old_val = rows[ri][ci]
            old_row = rows[ri]
            old_diffs = row_diff_sets[ri]

            # Generate candidate: perturb by a small random amount
            # Bias toward smaller values to reduce scope
            max_shift = max(5, int(0.1 * current_scope))
            shift = random.randint(-max_shift, max_shift // 2)  # bias downward
            nv = old_val + shift

            # Enforce ordering constraints
            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else current_scope
            if nv < lower:
                nv = lower
            if nv > upper:
                nv = upper
            if nv == old_val:
                continue

            new_row = old_row[:]
            new_row[ci] = nv
            new_diffs = row_diffs_set(new_row)

            # Check internal consistency (no duplicate diffs within row)
            if len(new_diffs) != diffs_per_row:
                continue

            # Check no collision with other rows
            other_diffs = all_diffs - old_diffs
            if new_diffs & other_diffs:
                continue

            # Valid move! Compute new scope
            new_max = max(new_row)
            new_scope = max(current_scope, new_max)

            # Acceptance criterion
            if new_scope <= current_scope:
                # Always accept improvements or equal
                accept = True
            else:
                delta = new_scope - current_scope
                if T > 0:
                    accept_prob = math.exp(-delta / T)
                    accept = random.random() < accept_prob
                else:
                    accept = False

            if accept:
                # Update state
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diff_sets[ri] = new_diffs
                rows[ri] = new_row
                current_scope = new_scope

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    stuck_count = 0
                else:
                    stuck_count += 1

                # Reheat if stuck
                if stuck_count > max_stuck:
                    stuck_count = 0
                    # Reset temperature by effectively restarting the schedule
                    # by doing nothing special - the frac-based T will keep going
                    pass
            else:
                stuck_count += 1

            # Early exit if we hit the counting bound
            if best_scope <= 112:
                break

            # Periodic check
            if total_iters % 5000 == 0 and time.time() > time_limit:
                break

        return best_rows, best_scope

    def local_search(rows, max_iters, time_limit):
        """Fast local search: only accept scope-decreasing moves."""
        rows = [r[:] for r in rows]
        best = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)

        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s

        it = 0
        while it < max_iters and time.time() < time_limit:
            it += 1
            ri = random.randrange(n)
            ci = random.randrange(1, L)

            old_val = rows[ri][ci]
            old_row = rows[ri]
            old_diffs = row_diff_sets[ri]

            candidates = set()
            for d in range(-10, 11):
                nv = old_val + d
                if nv >= 1:
                    candidates.add(nv)
            for _ in range(4):
                nv = random.randint(1, best_scope)
                candidates.add(nv)

            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope
            candidates = [c for c in candidates if lower <= c <= upper]
            if not candidates:
                continue

            random.shuffle(candidates)
            for nv in candidates:
                if nv == old_val:
                    continue
                new_row = old_row[:]
                new_row[ci] = nv
                new_diffs = row_diffs_set(new_row)

                if len(new_diffs) != diffs_per_row:
                    continue

                other_diffs = all_diffs - old_diffs
                if new_diffs & other_diffs:
                    continue

                new_scope = max(best_scope, max(new_row))
                if new_scope < best_scope:
                    all_diffs = (all_diffs - old_diffs) | new_diffs
                    row_diff_sets[ri] = new_diffs
                    rows[ri] = new_row
                    best = [r[:] for r in rows]
                    best_scope = new_scope
                    break

        return best, best_scope

    # ---------- Main search ----------
    # Start with pure greedy
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)

    # Phase 1: Randomized greedy restarts to find diverse starting points
    candidates_pool = [(best_scope, best_rows)]
    attempts = 0
    while time.time() < deadline - 50.0 and attempts < 60:
        attempts += 1
        rows = randomized_greedy(window=40 + random.randint(0, 80))
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            candidates_pool.insert(0, (sc, rows))
            if best_scope <= 112:
                break
        elif len(candidates_pool) < 8 and sc < best_scope + 30:
            candidates_pool.append((sc, rows))

    # Sort pool by scope
    candidates_pool.sort(key=lambda x: x[0])

    # Phase 2: Simulated annealing on top candidates
    for sc, start_rows in candidates_pool[:4]:
        if time.time() >= deadline - 15.0:
            break
        improved, new_sc = simulated_annealing(start_rows, time_limit=deadline - 15.0)
        if new_sc < best_scope:
            best_rows = improved
            best_scope = new_sc
            if best_scope <= 112:
                break

    # Phase 3: Local search polish
    if time.time() < deadline - 5.0:
        improved, sc = local_search(best_rows, max_iters=10000, time_limit=deadline - 5.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc

    return best_rows
