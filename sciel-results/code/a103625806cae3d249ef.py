import time
import random
import math
import sys

sys.setrecursionlimit(10000)


def construct(n_rows, k):
    """Construct a difference triangle set with n_rows rows of k+1 increasing
    integers starting at 0, minimizing the maximum entry."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # ---- Greedy fallback (always valid) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ri)
        return diffs

    def _all_used(rows):
        all_used = set()
        for row in rows:
            all_used |= _row_diffs(row)
        return all_used

    def _global_max(rows):
        return max(max(row) for row in rows)

    # ---- Simulated Annealing with structural moves ----
    def simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
        current_rows = [r[:] for r in initial_rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]

        best_rows = [r[:] for r in current_rows]
        best_val = current_val

        T = 500.0
        T_min = 0.5
        cool = 0.99997

        no_improve = 0
        max_no_improve = 80000
        restarts = 0
        max_restarts = 50

        # Adaptive step scale
        step_scale = 1.0

        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]

        def _try_single_move(ri, pi, new_val):
            """Try to set current_rows[ri][pi] = new_val. Returns (success, new_diffs, old_diffs)."""
            nonlocal all_used, row_diffs, row_maxes, current_val

            old_val = current_rows[ri][pi]
            if new_val == old_val:
                return False, None, None

            # Check bounds
            if new_val <= current_rows[ri][pi - 1]:
                return False, None, None
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                return False, None, None

            # Compute new diffs for this row
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            new_diffs = _row_diffs(cand_row)

            # Check internal consistency
            if len(new_diffs) != len(cand_row) * (len(cand_row) - 1) // 2:
                return False, None, None

            # Check cross-row conflicts: new diffs not in old row_diffs must not be in all_used
            old_rd = row_diffs[ri]
            for d in new_diffs:
                if d not in old_rd and d in all_used:
                    return False, None, None

            # Check that all old diffs are still covered or removed
            # Actually we just need: new_diffs ∪ (all_used - old_rd) has no conflicts
            # The above check suffices: any new diff not previously in this row must be free

            # Update
            all_used -= old_rd
            all_used |= new_diffs
            current_rows[ri] = cand_row
            row_diffs[ri] = new_diffs
            row_maxes[ri] = max(cand_row)
            current_val = max(row_maxes)

            return True, new_diffs, old_rd

        def _try_swap_rows(ri, rj):
            """Swap two entire rows. Returns success."""
            nonlocal all_used, row_diffs, row_maxes, current_val

            # Since all diffs are global, swapping rows doesn't change all_used
            # But it changes which row has which max
            current_rows[ri], current_rows[rj] = current_rows[rj], current_rows[ri]
            row_diffs[ri], row_diffs[rj] = row_diffs[rj], row_diffs[ri]
            row_maxes[ri], row_maxes[rj] = row_maxes[rj], row_maxes[ri]
            current_val = max(row_maxes)
            return True

        def _try_row_shift(ri, shift):
            """Shift all elements (except 0) of row ri by 'shift'. Returns success."""
            nonlocal all_used, row_diffs, row_maxes, current_val

            old_row = current_rows[ri]
            new_row = [0] + [x + shift for x in old_row[1:]]

            # Check strictly increasing
            for i in range(1, len(new_row)):
                if new_row[i] <= new_row[i - 1]:
                    return False

            new_diffs = _row_diffs(new_row)

            # Check conflicts
            old_rd = row_diffs[ri]
            for d in new_diffs:
                if d not in old_rd and d in all_used:
                    return False

            all_used -= old_rd
            all_used |= new_diffs
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            row_maxes[ri] = max(new_row)
            current_val = max(row_maxes)
            return True

        while time.time() < deadline and no_improve < max_no_improve:
            # Choose move type
            move_type = random.random()
            accepted = False

            if move_type < 0.55:
                # Single element move
                max_r = _max_rows()
                if max_r and random.random() < 0.65:
                    ri = random.choice(max_r)
                else:
                    ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)

                old_val = current_rows[ri][pi]

                # Generate candidates
                dec_range = int(min(80 * step_scale, max(0, old_val - current_rows[ri][pi - 1] - 1)))
                inc_range = int(min(30 * step_scale, max(0, (current_rows[ri][pi + 1] - 1 if pi < k else current_val + 40) - old_val)))

                candidates = []
                for d in range(1, dec_range + 1):
                    candidates.append(old_val - d)
                for d in range(1, inc_range + 1):
                    candidates.append(old_val + d)

                # Random jump
                if random.random() < 0.3:
                    low = current_rows[ri][pi - 1] + 1
                    high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
                    if low <= high:
                        candidates.append(random.randint(low, high))

                if candidates:
                    # Sort: decreasing first
                    candidates.sort(key=lambda v: (v > old_val, v))

                    for new_val in candidates:
                        if new_val <= current_rows[ri][pi - 1]:
                            continue
                        if pi < k and new_val >= current_rows[ri][pi + 1]:
                            continue

                        # Snapshot for rollback
                        snap_rows = [current_rows[ri][:]]
                        snap_used = set(all_used)
                        snap_rd = set(row_diffs[ri])
                        snap_rmax = row_maxes[ri]
                        snap_cval = current_val

                        success, _, _ = _try_single_move(ri, pi, new_val)
                        if not success:
                            continue

                        delta = current_val - snap_cval

                        # Acceptance
                        do_accept = False
                        if delta < 0:
                            do_accept = True
                        elif delta == 0:
                            if random.random() < 0.2:
                                do_accept = True
                        else:
                            if T > T_min:
                                prob = math.exp(-delta / max(T, 0.05))
                                if random.random() < prob:
                                    do_accept = True

                        if do_accept:
                            accepted = True
                            break
                        else:
                            # Rollback
                            current_rows[ri] = snap_rows[0]
                            all_used = snap_used
                            row_diffs[ri] = snap_rd
                            row_maxes[ri] = snap_rmax
                            current_val = snap_cval

            elif move_type < 0.75:
                # Swap two rows
                ri = random.randrange(n_rows)
                rj = random.randrange(n_rows)
                if ri != rj:
                    snap_rows = [current_rows[ri][:], current_rows[rj][:]]
                    snap_rd = [set(row_diffs[ri]), set(row_diffs[rj])]
                    snap_rmax = [row_maxes[ri], row_maxes[rj]]
                    snap_cval = current_val

                    _try_swap_rows(ri, rj)

                    delta = current_val - snap_cval
                    do_accept = False
                    if delta < 0:
                        do_accept = True
                    elif delta == 0:
                        if random.random() < 0.15:
                            do_accept = True
                    else:
                        if T > T_min:
                            prob = math.exp(-delta / max(T, 0.05))
                            if random.random() < prob:
                                do_accept = True

                    if do_accept:
                        accepted = True
                    else:
                        current_rows[ri], current_rows[rj] = snap_rows[0], snap_rows[1]
                        row_diffs[ri], row_diffs[rj] = snap_rd[0], snap_rd[1]
                        row_maxes[ri], row_maxes[rj] = snap_rmax[0], snap_rmax[1]
                        current_val = snap_cval

            else:
                # Row shift
                ri = random.randrange(n_rows)
                # Small shifts
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                if random.random() < 0.2:
                    shift = random.randint(-10, 10)

                snap_rows = [current_rows[ri][:]]
                snap_used = set(all_used)
                snap_rd = set(row_diffs[ri])
                snap_rmax = row_maxes[ri]
                snap_cval = current_val

                if _try_row_shift(ri, shift):
                    delta = current_val - snap_cval
                    do_accept = False
                    if delta < 0:
                        do_accept = True
                    elif delta == 0:
                        if random.random() < 0.15:
                            do_accept = True
                    else:
                        if T > T_min:
                            prob = math.exp(-delta / max(T, 0.05))
                            if random.random() < prob:
                                do_accept = True

                    if do_accept:
                        accepted = True
                    else:
                        current_rows[ri] = snap_rows[0]
                        all_used = snap_used
                        row_diffs[ri] = snap_rd
                        row_maxes[ri] = snap_rmax
                        current_val = snap_cval

            if accepted:
                no_improve = 0
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
            else:
                no_improve += 1

            # Cooling
            T = max(T_min, T * cool)

            # Adaptive step
            if no_improve > 10000:
                step_scale = max(0.5, step_scale * 0.95)
            elif no_improve < 1000:
                step_scale = min(2.0, step_scale * 1.01)

            # Restart if stuck
            if no_improve > max_no_improve and restarts < max_restarts:
                restarts += 1
                # Perturb current solution
                for _ in range(random.randint(5, 15)):
                    ri = random.randrange(n_rows)
                    pi = random.randrange(1, k + 1)
                    old_val = current_rows[ri][pi]
                    low = current_rows[ri][pi - 1] + 1
                    high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
                    if low <= high:
                        new_val = random.randint(low, high)
                        snap_rows = [current_rows[ri][:]]
                        snap_used = set(all_used)
                        snap_rd = set(row_diffs[ri])
                        snap_rmax = row_maxes[ri]
                        snap_cval = current_val
                        if _try_single_move(ri, pi, new_val):
                            pass
                        else:
                            current_rows[ri] = snap_rows[0]
                            all_used = snap_used
                            row_diffs[ri] = snap_rd
                            row_maxes[ri] = snap_rmax
                            current_val = snap_cval

                no_improve = 0
                T = max(T, 200.0)

        return best_rows, best_val

    # ---- Local search polish ----
    def local_search_polish(rows, n_rows, k, deadline):
        """Greedy local search to minimize max entry."""
        best_rows = [r[:] for r in rows]
        used_diffs = _all_used(best_rows)

        while time.time() < deadline - 3:
            progress = False
            row_order = list(range(n_rows))
            random.shuffle(row_order)

            for r_idx in row_order:
                if time.time() > deadline - 3:
                    break

                row = best_rows[r_idx]
                row_rd = _row_diffs(row)

                for pos in range(k, 0, -1):
                    old_val = row[pos]

                    diffs_involving_old = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            diffs_involving_old.add(old_val - row[i])
                        else:
                            diffs_involving_old.add(row[i] - old_val)

                    forbidden_diffs = used_diffs - diffs_involving_old

                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1
                    if pos + 1 < len(row):
                        upper_bound = min(upper_bound, row[pos + 1] - 1)

                    if upper_bound < lower_bound:
                        continue

                    found = None
                    for new_val in range(lower_bound, upper_bound + 1):
                        valid = True
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val

                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)

                        if valid:
                            found = new_val
                            break

                    if found is not None and found < old_val:
                        used_diffs -= diffs_involving_old
                        used_diffs |= new_diffs
                        row[pos] = found
                        progress = True

            if not progress:
                break

        return best_rows

    # ---- Main loop: multiple SA runs with different seeds ----
    num_sa_runs = 4
    sa_time_each = (deadline - start_time - 5) / max(num_sa_runs, 1)

    for sa_run in range(num_sa_runs):
        if time.time() >= deadline - 8:
            break

        # For subsequent runs, perturb the best solution
        if sa_run == 0:
            init = best_solution
        else:
            init = [r[:] for r in best_solution]
            # Perturb
            for _ in range(random.randint(3, 8)):
                ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)
                old_val = init[ri][pi]
                low = init[ri][pi - 1] + 1
                high = (init[ri][pi + 1] - 1) if pi < k else max(init[ri]) + 20
                if low <= high:
                    new_val = random.randint(low, high)
                    # Validate quickly
                    cand_row = init[ri][:]
                    cand_row[pi] = new_val
                    rd = _row_diffs(cand_row)
                    if len(rd) == len(cand_row) * (len(cand_row) - 1) // 2:
                        all_d = _all_used(init)
                        old_rd = _row_diffs(init[ri])
                        conflict = False
                        for d in rd:
                            if d not in old_rd and d in all_d:
                                conflict = True
                                break
                        if not conflict:
                            init[ri] = cand_row

        sa_deadline = min(deadline - 8, start_time + sa_run * sa_time_each + sa_time_each)
        sa_result, sa_scope = simulated_annealing(init, n_rows, k, start_time, sa_deadline)

        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_result

    # Final polish
    if time.time() < deadline - 2:
        polished = local_search_polish(best_solution, n_rows, k, deadline)
        if polished:
            curr_scope = max(max(r) for r in polished)
            if curr_scope < best_scope:
                best_scope = curr_scope
                best_solution = polished

    return best_solution
