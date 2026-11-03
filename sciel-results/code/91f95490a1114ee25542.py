<<<<<<< SEARCH
# ---------------------------------------------------------------------------
# Phase 2 – Simulated annealing in difference-space
# ---------------------------------------------------------------------------

def _sa_difference_space(rows, k, deadline, rng):
    """
    SA where the state is (rows, diff_counts, row_masks).
    Moves:
      1. Perturb one consecutive difference of a random row.
      2. Rebuild one row via randomised greedy (with the other rows fixed).
    Cost = scope + 0.005 * sum_entries + 10000 * collisions.
    """
    n = len(rows)
    cur_rows = [list(r) for r in rows]
    cur_consec = [[r[i + 1] - r[i] for i in range(len(r) - 1)] for r in cur_rows]
    cur_masks = [_row_mask(r) for r in cur_rows]
    cur_scope = _scope(cur_rows)
    cur_sum = _sum_entries(cur_rows)

    # diff_counts: dict diff -> count
    diff_counts = {}
    for r in cur_rows:
        m = _row_mask(r)
        # iterate set bits
        mm = m
        while mm:
            d = (mm & (-mm)).bit_length() - 1
            diff_counts[d] = diff_counts.get(d, 0) + 1
            mm &= mm - 1

    def _collisions(dc):
        return sum(max(0, c - 1) for c in dc.values())

    def _cost(sc, sm, dc):
        return sc + 0.005 * sm + 10000.0 * _collisions(dc)

    best_rows = [list(r) for r in cur_rows]
    best_consec = [c[:] for c in cur_consec]
    best_masks = cur_masks[:]
    best_scope = cur_scope
    best_sum = cur_sum
    best_dc = dict(diff_counts)
    best_cost = _cost(cur_scope, cur_sum, diff_counts)

    T = 5.0
    T_min = 0.05
    cooling = 0.9997

    t0 = time.time()
    while time.time() < deadline:
        if T < T_min:
            T = 5.0
            # small perturbation: random consecutive diff
            ri = rng.randrange(n)
            p = rng.randrange(len(cur_consec[ri]))
            cur_consec[ri][p] = max(1, cur_consec[ri][p] + rng.randint(-2, 2))
            # rebuild row from consec
            row = [0]
            for d in cur_consec[ri]:
                row.append(row[-1] + d)
            cur_rows[ri] = row
            cur_masks[ri] = _row_mask(row)
            # update diff_counts
            old_dc = dict(diff_counts)
            # remove old row diffs
            old_m = best_masks[ri]  # not quite right; use cur before update
            # Actually we need to recompute. Let's just rebuild diff_counts.
            diff_counts = {}
            for r in cur_rows:
                m = _row_mask(r)
                mm = m
                while mm:
                    d = (mm & (-mm)).bit_length() - 1
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    mm &= mm - 1
            cur_scope = _scope(cur_rows)
            cur_sum = _sum_entries(cur_rows)
            cur_cost = _cost(cur_scope, cur_sum, diff_counts)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_scope = cur_scope
                best_sum = cur_sum
                best_rows = [list(r) for r in cur_rows]
                best_consec = [c[:] for c in cur_consec]
                best_masks = cur_masks[:]
                best_dc = dict(diff_counts)
            continue

        r = rng.random()
        if r < 0.70:
            # Perturb one consecutive difference
            ri = rng.randrange(n)
            p = rng.randrange(len(cur_consec[ri]))
            old_d = cur_consec[ri][p]
            new_d = max(1, old_d + rng.randint(-3, 3))
            if new_d == old_d:
                T *= cooling
                continue
            cur_consec[ri][p] = new_d
            # rebuild row
            row = [0]
            for d in cur_consec[ri]:
                row.append(row[-1] + d)
            new_mask = _row_mask(row)

            # quick collision check: new_mask & (used_without_ri)
            used_without = 0
            for j in range(n):
                if j != ri:
                    used_without |= cur_masks[j]
            if new_mask & used_without:
                # collision – reject
                cur_consec[ri][p] = old_d
                T *= cooling
                continue

            # also check internal duplicates: row must have distinct pairwise diffs
            # (already guaranteed by mask having popcount = k(k+1)/2)
            expected_popcount = k * (k + 1) // 2
            if new_mask.bit_count() != expected_popcount:
                cur_consec[ri][p] = old_d
                T *= cooling
                continue

            cur_rows[ri] = row
            cur_masks[ri] = new_mask
            cur_scope = _scope(cur_rows)
            cur_sum = _sum_entries(cur_rows)

            # update diff_counts incrementally: remove old, add new
            # old row mask
            old_row = [0]
            for d in cur_consec[ri]:
                old_row.append(old_row[-1] + d)
            # Wait, we already changed cur_consec. Let's rebuild diff_counts.
            diff_counts = {}
            for rr in cur_rows:
                m = _row_mask(rr)
                mm = m
                while mm:
                    d = (mm & (-mm)).bit_length() - 1
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    mm &= mm - 1

            new_cost = _cost(cur_scope, cur_sum, diff_counts)
            dE = new_cost - _cost(best_scope, best_sum, best_dc)
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, T_min)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = cur_scope
                    best_sum = cur_sum
                    best_rows = [list(r) for r in cur_rows]
                    best_consec = [c[:] for c in cur_consec]
                    best_masks = cur_masks[:]
                    best_dc = dict(diff_counts)
            else:
                # revert
                cur_consec[ri][p] = old_d
                row = [0]
                for d in cur_consec[ri]:
                    row.append(row[-1] + d)
                cur_rows[ri] = row
                cur_masks[ri] = _row_mask(row)
                diff_counts = {}
                for rr in cur_rows:
                    m = _row_mask(rr)
                    mm = m
                    while mm:
                        d = (mm & (-mm)).bit_length() - 1
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                        mm &= mm - 1
        else:
            # Rebuild one row via randomised greedy
            ri = rng.randrange(n)
            used_without = 0
            for j in range(n):
                if j != ri:
                    used_without |= cur_masks[j]

            # randomised greedy rebuild
            row = [0]
            consec = []
            local_mask = 0
            success = True
            for _j in range(k):
                found = False
                start = 1
                if rng.random() < 0.3:
                    start = rng.randint(1, 4)
                for d in range(start, cur_scope + 10):
                    new_entry = row[-1] + d
                    new_mask = 0
                    for a in row:
                        new_mask |= (1 << (new_entry - a))
                    if (new_mask & used_without) or (new_mask & local_mask):
                        continue
                    row.append(new_entry)
                    consec.append(d)
                    for a in row[:-1]:
                        local_mask |= (1 << (row[-1] - a))
                    found = True
                    break
                if not found:
                    success = False
                    break
            if not success:
                T *= cooling
                continue

            new_mask = _row_mask(row)
            if new_mask & used_without:
                T *= cooling
                continue
            if new_mask.bit_count() != k * (k + 1) // 2:
                T *= cooling
                continue

            cur_rows[ri] = row
            cur_consec[ri] = consec
            cur_masks[ri] = new_mask
            cur_scope = _scope(cur_rows)
            cur_sum = _sum_entries(cur_rows)
            diff_counts = {}
            for rr in cur_rows:
                m = _row_mask(rr)
                mm = m
                while mm:
                    d = (mm & (-mm)).bit_length() - 1
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    mm &= mm - 1
            new_cost = _cost(cur_scope, cur_sum, diff_counts)
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = cur_scope
                best_sum = cur_sum
                best_rows = [list(r) for r in cur_rows]
                best_consec = [c[:] for c in cur_consec]
                best_masks = cur_masks[:]
                best_dc = dict(diff_counts)

        T *= cooling

    return best_rows, best_scope
=======
# ---------------------------------------------------------------------------
# Phase 2 – Simulated annealing in difference-space
# ---------------------------------------------------------------------------

def _sa_difference_space(rows, k, deadline, rng):
    """
    SA where the state is (rows, diff_counts, row_masks).
    Moves:
      1. Perturb one consecutive difference of a random row.
      2. Rebuild one row via randomised greedy (with the other rows fixed).
    Cost = scope + 0.005 * sum_entries + 10000 * collisions.
    Uses incremental diff_counts updates for speed.
    """
    n = len(rows)
    cur_rows = [list(r) for r in rows]
    cur_consec = [[r[i + 1] - r[i] for i in range(len(r) - 1)] for r in cur_rows]
    cur_masks = [_row_mask(r) for r in cur_rows]
    cur_scope = _scope(cur_rows)
    cur_sum = _sum_entries(cur_rows)

    # diff_counts: dict diff -> count
    diff_counts = _build_diff_counts(cur_rows)
    cur_collisions = sum(max(0, c - 1) for c in diff_counts.values())

    def _cost(sc, sm, coll):
        return sc + 0.005 * sm + 10000.0 * coll

    best_rows = [list(r) for r in cur_rows]
    best_consec = [c[:] for c in cur_consec]
    best_masks = cur_masks[:]
    best_scope = cur_scope
    best_sum = cur_sum
    best_dc = dict(diff_counts)
    best_collisions = cur_collisions
    best_cost = _cost(cur_scope, cur_sum, cur_collisions)

    T = 5.0
    T_min = 0.05
    cooling = 0.9997
    expected_popcount = k * (k + 1) // 2

    t0 = time.time()
    while time.time() < deadline:
        if T < T_min:
            T = 5.0
            # Restart with perturbation: pick a random row and perturb a diff
            ri = rng.randrange(n)
            p = rng.randrange(len(cur_consec[ri]))
            old_d = cur_consec[ri][p]
            cur_consec[ri][p] = max(1, old_d + rng.randint(-3, 3))
            old_row = cur_rows[ri]
            row = [0]
            for d in cur_consec[ri]:
                row.append(row[-1] + d)
            new_mask = _row_mask(row)
            if new_mask.bit_count() != expected_popcount:
                cur_consec[ri][p] = old_d
                continue
            used_without = 0
            for j in range(n):
                if j != ri:
                    used_without |= cur_masks[j]
            if new_mask & used_without:
                cur_consec[ri][p] = old_d
                continue
            _update_diff_counts(diff_counts, old_row, row)
            cur_collisions = sum(max(0, c - 1) for c in diff_counts.values())
            cur_rows[ri] = row
            cur_masks[ri] = new_mask
            cur_scope = _scope(cur_rows)
            cur_sum = _sum_entries(cur_rows)
            cur_cost = _cost(cur_scope, cur_sum, cur_collisions)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_scope = cur_scope
                best_sum = cur_sum
                best_rows = [list(r) for r in cur_rows]
                best_consec = [c[:] for c in cur_consec]
                best_masks = cur_masks[:]
                best_dc = dict(diff_counts)
                best_collisions = cur_collisions
            continue

        r = rng.random()
        if r < 0.70:
            # Perturb one consecutive difference
            ri = rng.randrange(n)
            p = rng.randrange(len(cur_consec[ri]))
            old_d = cur_consec[ri][p]
            new_d = max(1, old_d + rng.randint(-3, 3))
            if new_d == old_d:
                T *= cooling
                continue
            cur_consec[ri][p] = new_d
            # rebuild row
            row = [0]
            for d in cur_consec[ri]:
                row.append(row[-1] + d)
            new_mask = _row_mask(row)

            # quick collision check: new_mask & (used_without_ri)
            used_without = 0
            for j in range(n):
                if j != ri:
                    used_without |= cur_masks[j]
            if new_mask & used_without:
                cur_consec[ri][p] = old_d
                T *= cooling
                continue

            if new_mask.bit_count() != expected_popcount:
                cur_consec[ri][p] = old_d
                T *= cooling
                continue

            old_row = cur_rows[ri]
            _update_diff_counts(diff_counts, old_row, row)
            new_collisions = sum(max(0, c - 1) for c in diff_counts.values())
            cur_rows[ri] = row
            cur_masks[ri] = new_mask
            cur_scope = _scope(cur_rows)
            cur_sum = _sum_entries(cur_rows)

            new_cost = _cost(cur_scope, cur_sum, new_collisions)
            best_cost_now = _cost(best_scope, best_sum, best_collisions)
            dE = new_cost - best_cost_now
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, T_min)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = cur_scope
                    best_sum = cur_sum
                    best_rows = [list(r) for r in cur_rows]
                    best_consec = [c[:] for c in cur_consec]
                    best_masks = cur_masks[:]
                    best_dc = dict(diff_counts)
                    best_collisions = new_collisions
            else:
                # revert
                cur_consec[ri][p] = old_d
                _update_diff_counts(diff_counts, row, old_row)
                cur_rows[ri] = old_row
                cur_masks[ri] = _row_mask(old_row)
        else:
            # Rebuild one row via randomised greedy
            ri = rng.randrange(n)
            used_without = 0
            for j in range(n):
                if j != ri:
                    used_without |= cur_masks[j]

            # randomised greedy rebuild
            row = [0]
            consec = []
            local_mask = 0
            success = True
            for _j in range(k):
                found = False
                start = 1
                if rng.random() < 0.3:
                    start = rng.randint(1, 4)
                for d in range(start, cur_scope + 10):
                    new_entry = row[-1] + d
                    new_mask = 0
                    for a in row:
                        new_mask |= (1 << (new_entry - a))
                    if (new_mask & used_without) or (new_mask & local_mask):
                        continue
                    row.append(new_entry)
                    consec.append(d)
                    for a in row[:-1]:
                        local_mask |= (1 << (row[-1] - a))
                    found = True
                    break
                if not found:
                    success = False
                    break
            if not success:
                T *= cooling
                continue

            new_mask = _row_mask(row)
            if new_mask & used_without:
                T *= cooling
                continue
            if new_mask.bit_count() != expected_popcount:
                T *= cooling
                continue

            old_row = cur_rows[ri]
            _update_diff_counts(diff_counts, old_row, row)
            new_collisions = sum(max(0, c - 1) for c in diff_counts.values())
            cur_rows[ri] = row
            cur_consec[ri] = consec
            cur_masks[ri] = new_mask
            cur_scope = _scope(cur_rows)
            cur_sum = _sum_entries(cur_rows)
            new_cost = _cost(cur_scope, cur_sum, new_collisions)
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = cur_scope
                best_sum = cur_sum
                best_rows = [list(r) for r in cur_rows]
                best_consec = [c[:] for c in cur_consec]
                best_masks = cur_masks[:]
                best_dc = dict(diff_counts)
                best_collisions = new_collisions

        T *= cooling

    return best_rows, best_scope
>>>>>>> REPLACE

