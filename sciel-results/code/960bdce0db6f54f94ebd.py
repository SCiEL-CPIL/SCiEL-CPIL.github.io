import time
import random
import math
import sys
import itertools

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Hybrid: greedy init -> SA local search with incremental diff tracking ->
    row-level re-optimization (local moves + greedy/backtracking rebuild) ->
    smooth soft-max (log-sum-exp) surrogate polish -> final high-precision pass.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

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

    def local_search_init(n_rows, k, seed=None):
        """Greedy initialization (deterministic given used-set ordering)."""
        if seed is not None:
            random.seed(seed)
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

    def _row_reopt_bt(row_init, all_used_excl, k, max_scope, max_depth=3):
        """
        Rebuild a single row from scratch with greedy fill + limited backtracking,
        constrained to max_scope and avoiding all_used_excl (differences of other rows).
        Returns a valid row or None.
        """
        n = k + 1

        def _try_fill(row, pos, used_local, depth):
            if pos == n:
                return row[:]
            start = row[pos - 1] + 1
            end = max_scope + 1
            for v in range(start, end):
                new_diffs = {v - row[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue
                row[pos] = v
                used_local_new = used_local | new_diffs
                if depth < max_depth:
                    result = _try_fill(row, pos + 1, used_local_new, depth + 1)
                    if result is not None:
                        return result
                else:
                    rest_row = row[:]
                    rest_used = used_local_new
                    for p in range(pos + 1, n):
                        found = False
                        for v2 in range(rest_row[p - 1] + 1, max_scope + 1):
                            nd2 = {v2 - rest_row[i] for i in range(p)}
                            if len(nd2) != p:
                                continue
                            if any(d in all_used_excl for d in nd2):
                                continue
                            if any(d in rest_used for d in nd2):
                                continue
                            rest_row.append(v2)
                            rest_used |= nd2
                            found = True
                            break
                        if not found:
                            return None
                    return rest_row
            return None

        row = [0] + [0] * k
        return _try_fill(row, 1, set(), 0)

    def _reoptimize_row(row, other_diffs, k, max_scope=None, iters=60, neutral_prob=0.25):
        """
        Globally re-optimize a single row against fixed differences from other rows.
        1) local moves with O(1) conflict checks via diff-set algebra
        2) if stuck, try a from-scratch greedy/backtracking rebuild within max_scope
        Returns the improved row.
        """
        n = k + 1
        best_row = list(row)
        best_max = max(best_row)

        if max_scope is not None and best_max > max_scope:
            return best_row

        other_set = set(other_diffs)

        def _row_diff_set(r):
            ds = set()
            for i in range(n):
                ri = r[i]
                for j in range(i + 1, n):
                    ds.add(r[j] - ri)
            return ds

        cur_diffs = _row_diff_set(best_row)

        for _ in range(iters):
            improved = False
            for pi in range(1, n):
                old_val = best_row[pi]
                min_allowed = best_row[pi - 1] + 1
                if pi < n - 1:
                    max_allowed = best_row[pi + 1] - 1
                else:
                    max_allowed = (max_scope - 1) if max_scope else old_val

                # Try decreasing values first (primary optimization direction)
                for new_val in range(old_val - 1, min_allowed - 1, -1):
                    if new_val < min_allowed:
                        break
                    if pi < n - 1 and new_val >= best_row[pi + 1]:
                        break

                    removed = {old_val - best_row[i] for i in range(pi)}
                    removed |= {best_row[j] - old_val for j in range(pi + 1, n)}
                    added = {new_val - best_row[i] for i in range(pi)}
                    added |= {best_row[j] - new_val for j in range(pi + 1, n)}

                    cur_minus_removed = cur_diffs - removed
                    if added & cur_minus_removed:
                        continue
                    if added & other_set:
                        continue

                    cand = best_row[:]
                    cand[pi] = new_val
                    new_max = max(cand)

                    if new_max < best_max:
                        best_row = cand
                        best_max = new_max
                        cur_diffs = (cur_diffs - removed) | added
                        improved = True
                        break
                    elif new_max == best_max:
                        if random.random() < neutral_prob:
                            best_row = cand
                            cur_diffs = (cur_diffs - removed) | added
                            improved = True
                            break
            if not improved:
                break

        # Fallback: from-scratch rebuild within max_scope (may shrink further)
        if max_scope is not None and max_scope < 400:
            rebuilt = _row_reopt_bt(best_row, other_set, k, max_scope, max_depth=3)
            if rebuilt is not None and max(rebuilt) < best_max:
                best_row = rebuilt
                best_max = max(best_row)

        return best_row

    def simulated_annealing(rows, n_rows, k, start_time, deadline):
        """SA-based local search with row-level re-optimization to minimize the max entry."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]

        best_rows = [r[:] for r in current_rows]
        best_val = current_val

        T = 500.0
        T_min = 0.5
        cool = 0.99998

        no_improve = 0
        max_no_improve = 30000
        restarts = 0
        max_restarts = 20
        last_reopt_time = time.time()
        reopt_interval = 1.5

        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]

        def _other_diffs_except(ri):
            other = set()
            for j in range(n_rows):
                if j != ri:
                    other |= row_diffs[j]
            return other

        while time.time() < deadline and no_improve < max_no_improve:
            now = time.time()
            if now - last_reopt_time > reopt_interval and no_improve > 300:
                last_reopt_time = now
                target_row = max(range(n_rows), key=lambda i: row_maxes[i])
                other_diffs = _other_diffs_except(target_row)
                improved_row = _reoptimize_row(
                    current_rows[target_row], other_diffs, k,
                    max_scope=best_val, iters=40
                )
                if max(improved_row) < row_maxes[target_row]:
                    new_diffs = _row_diffs(improved_row)
                    all_used -= row_diffs[target_row]
                    all_used |= new_diffs
                    current_rows[target_row] = improved_row
                    row_diffs[target_row] = new_diffs
                    row_maxes[target_row] = max(improved_row)
                    current_val = max(row_maxes)
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    continue

            max_r = _max_rows()
            if max_r and random.random() < 0.6:
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)

            old_val = current_rows[ri][pi]

            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(100, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)

            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
            max_increase = min(30, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)

            if random.random() < 0.3:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
                if low <= high:
                    candidates.append(random.randint(low, high))

            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue

            candidates.sort(key=lambda v: (v > old_val, v))

            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue

                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val

                new_diffs = set()
                valid = True
                n = len(cand_row)
                for i in range(n):
                    ri_val = cand_row[i]
                    for j in range(i + 1, n):
                        d = cand_row[j] - ri_val
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                if not valid:
                    continue

                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        valid = False
                        break

                if not valid:
                    continue

                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max

                delta = new_global_max - current_val

                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True

                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)

                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1

                    moved = True
                    break

            if not moved:
                no_improve += 1

            T = max(T_min, T * cool)

            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                all_used = _all_used(current_rows)
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = 300.0
                no_improve = 0
                last_reopt_time = time.time()

        return best_rows, best_val

    def soft_max_polish(rows, n_rows, k, deadline):
        """
        Smooth surrogate polish: minimize a soft-max (log-sum-exp) of row entries
        by greedily pushing each element down one step when it reduces the surrogate
        and stays valid. Complements the nonsmooth min/max objective.
        """
        current_rows = [r[:] for r in rows]
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]

        best_rows = [r[:] for r in current_rows]
        best_val = max(row_maxes)

        beta = 0.4  # inverse temperature for log-sum-exp

        def _softmax(rows):
            # sum over rows of logsumexp(beta * row_entries)
            total = 0.0
            for r in rows:
                m = max(r)
                s = sum(math.exp(beta * (v - m)) for v in r)
                total += m + math.log(s) / beta
            return total

        cur_soft = _softmax(current_rows)
        best_soft = cur_soft

        no_improve = 0
        max_no_improve = 20000

        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            # process rows in descending max order
            order = sorted(range(n_rows), key=lambda i: -row_maxes[i])
            for ri in order:
                if time.time() > deadline:
                    return best_rows, best_val
                row = current_rows[ri]
                n = len(row)
                for pi in range(1, n):
                    old_val = row[pi]
                    min_allowed = row[pi - 1] + 1
                    if old_val <= min_allowed:
                        continue
                    new_val = old_val - 1
                    if pi < n - 1 and new_val >= row[pi + 1]:
                        continue
                    cand = row[:]
                    cand[pi] = new_val
                    new_diffs = set()
                    valid = True
                    for i in range(n):
                        ri_val = cand[i]
                        for j in range(i + 1, n):
                            d = cand[j] - ri_val
                            if d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                    if not valid:
                        continue
                    for d in new_diffs:
                        if d in all_used and d not in row_diffs[ri]:
                            valid = False
                            break
                    if not valid:
                        continue

                    # compute new soft-max incrementally
                    old_row_soft = max(row) + math.log(
                        sum(math.exp(beta * (v - max(row))) for v in row)) / beta
                    new_row_soft = max(cand) + math.log(
                        sum(math.exp(beta * (v - max(cand))) for v in cand)) / beta
                    cand_soft = cur_soft - old_row_soft + new_row_soft

                    if cand_soft < cur_soft:
                        all_used -= row_diffs[ri]
                        all_used |= new_diffs
                        current_rows[ri] = cand
                        row_diffs[ri] = new_diffs
                        row_maxes[ri] = max(cand)
                        cur_soft = cand_soft
                        improved = True

            if improved:
                cur_max = max(row_maxes)
                if cur_max < best_val:
                    best_val = cur_max
                    best_rows = [r[:] for r in current_rows]
                    best_soft = cur_soft
                no_improve = 0
            else:
                no_improve += 1

        return best_rows, best_val

    def final_polish(rows, n_rows, k, deadline):
        """High-precision polish: tighter tolerance, more iterations, row re-optimization."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]

        best_rows = [r[:] for r in current_rows]
        best_val = current_val

        T = 100.0
        T_min = 0.1
        cool = 0.99995

        no_improve = 0
        max_no_improve = 50000

        while time.time() < deadline and no_improve < max_no_improve:
            target_row = max(range(n_rows), key=lambda i: row_maxes[i])
            other_diffs = set()
            for j in range(n_rows):
                if j != target_row:
                    other_diffs |= row_diffs[j]
            improved_row = _reoptimize_row(
                current_rows[target_row], other_diffs, k,
                max_scope=best_val, iters=80, neutral_prob=0.15
            )
            if max(improved_row) < row_maxes[target_row]:
                new_diffs = _row_diffs(improved_row)
                all_used -= row_diffs[target_row]
                all_used |= new_diffs
                current_rows[target_row] = improved_row
                row_diffs[target_row] = new_diffs
                row_maxes[target_row] = max(improved_row)
                current_val = max(row_maxes)
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                continue

            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)

            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1

            moved = False
            for d in range(1, min(20, old_val - min_allowed + 1)):
                new_val = old_val - d
                if new_val <= min_allowed:
                    break
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    break

                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val

                new_diffs = set()
                valid = True
                n = len(cand_row)
                for i in range(n):
                    ri_val = cand_row[i]
                    for j in range(i + 1, n):
                        dd = cand_row[j] - ri_val
                        if dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)

                if not valid:
                    continue

                for dd in new_diffs:
                    if dd in all_used and dd not in row_diffs[ri]:
                        valid = False
                        break

                if not valid:
                    continue

                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max

                delta = new_global_max - current_val

                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.1:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.05))
                        if random.random() < prob:
                            accept = True

                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)

                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1

                    moved = True
                    break

            if not moved:
                no_improve += 1

            T = max(T_min, T * cool)

        return best_rows, best_val

    # ---- Multi-start SA trials ----
    for trial in range(12):
        if time.time() > deadline - 5:
            break

        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)

        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]

        remaining = deadline - time.time()
        if trial == 0:
            trial_deadline = time.time() + min(remaining * 0.35, 45.0)
        elif trial == 1:
            trial_deadline = time.time() + min(remaining * 0.25, 35.0)
        elif trial == 2:
            trial_deadline = time.time() + min(remaining * 0.2, 25.0)
        else:
            trial_deadline = time.time() + min(remaining * 0.12, 15.0)

        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)

        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows

    # ---- Smooth soft-max surrogate polish ----
    if time.time() < deadline - 8:
        sm_deadline = time.time() + min((deadline - time.time()) * 0.30, 30.0)
        sm_rows, sm_scope = soft_max_polish(
            [r[:] for r in best_solution], n_rows, k, sm_deadline
        )
        if sm_scope < best_scope:
            best_scope = sm_scope
            best_solution = sm_rows

    # ---- Final high-precision polish ----
    if time.time() < deadline - 4:
        polish_deadline = min(deadline - 3, start_time + 143.0)
        polish_rows, polish_scope = final_polish(
            [r[:] for r in best_solution], n_rows, k, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows

    # ---- Final row-level re-optimization pass over ALL rows ----
    if time.time() < deadline - 1:
        for _pass in range(5):
            if time.time() > deadline - 1:
                break
            improved_any = False
            order = sorted(range(n_rows), key=lambda i: -max(best_solution[i]))
            for ri in order:
                if time.time() > deadline - 1:
                    break
                other_diffs = set()
                for j in range(n_rows):
                    if j != ri:
                        other_diffs |= _row_diffs(best_solution[j])
                improved_row = _reoptimize_row(
                    best_solution[ri], other_diffs, k,
                    max_scope=best_scope, iters=80, neutral_prob=0.15
                )
                if max(improved_row) < max(best_solution[ri]):
                    best_solution[ri] = improved_row
                    best_scope = max(max(row) for row in best_solution)
                    improved_any = True
            if not improved_any:
                break

    return best_solution
# EVOLVE-BLOCK-END
