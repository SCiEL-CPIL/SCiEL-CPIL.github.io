import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1          # k + 1 integers per row
    deadline = time.time() + 117.0
    n = n_rows
    L = row_len
    BIG = 1000000            # collision penalty weight (epigraph: collisions >> scope)

    def get_diffs(row):
        d = []
        for i in range(L):
            ai = row[i]
            for j in range(i + 1, L):
                d.append(row[j] - ai)
        return d

    # ---------- Greedy baseline ----------
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = max(r[-1] for r in best_rows)

    # ---------- Core incremental SA ----------
    # State: rows, row_diffs, diff_counts (dict d->count), collision_count, max_scope
    def make_state(rows):
        row_diffs = [get_diffs(r) for r in rows]
        diff_counts = {}
        collision_count = 0
        for rd in row_diffs:
            for d in rd:
                c = diff_counts.get(d, 0)
                if c == 1:
                    collision_count += 1
                diff_counts[d] = c + 1
        max_scope = max(r[-1] for r in rows)
        return rows, row_diffs, diff_counts, collision_count, max_scope

    def cost_of(collision_count, max_scope):
        return collision_count * BIG + max_scope

    # Apply a change of element (r_idx, i_idx) from old_val to new_val, updating state in place.
    # Returns the set of diffs that changed (for reverting).
    def apply_change(rows, row_diffs, diff_counts, collision_count, r_idx, i_idx, new_val):
        row = rows[r_idx]
        old_val = row[i_idx]
        if new_val == old_val:
            return None, None
        # Diffs removed (involving i_idx)
        removed = []
        for j in range(L):
            if j != i_idx:
                removed.append(abs(row[j] - old_val))
        # Update counts for removed
        for d in removed:
            c = diff_counts[d]
            if c == 2:
                collision_count -= 1
            c -= 1
            if c == 0:
                del diff_counts[d]
            else:
                diff_counts[d] = c
        # Apply
        row[i_idx] = new_val
        # Diffs added
        added = []
        for j in range(L):
            if j != i_idx:
                added.append(abs(row[j] - new_val))
        for d in added:
            c = diff_counts.get(d, 0)
            if c == 1:
                collision_count += 1
            diff_counts[d] = c + 1
        # Rebuild this row's diff list
        row_diffs[r_idx] = get_diffs(row)
        return (removed, added), None

    def revert_change(rows, row_diffs, diff_counts, collision_count, r_idx, i_idx, old_val, removed, added):
        row = rows[r_idx]
        # Remove 'added'
        for d in added:
            c = diff_counts[d]
            if c == 2:
                collision_count -= 1
            c -= 1
            if c == 0:
                del diff_counts[d]
            else:
                diff_counts[d] = c
        row[i_idx] = old_val
        # Re-add 'removed'
        for d in removed:
            c = diff_counts.get(d, 0)
            if c == 1:
                collision_count += 1
            diff_counts[d] = c + 1
        row_diffs[r_idx] = get_diffs(row)
        return collision_count

    def sa_optimize(initial_rows, time_limit, bound):
        rows, row_diffs, diff_counts, collision_count, max_scope = make_state(initial_rows)
        cur_cost = cost_of(collision_count, max_scope)

        best_cost = cur_cost
        best_rows = [r[:] for r in rows]
        best_max = max_scope

        T = 30.0
        cooling = 0.9999
        start = time.time()
        last_improve = start
        no_improve_steps = 0
        step = 0
        p_last = 0.55  # probability of moving the last element (scope-biased)

        while time.time() - start < time_limit:
            # Scope-biased element choice
            if random.random() < p_last:
                i_idx = L - 1
            else:
                i_idx = random.randint(1, L - 1)
            r_idx = random.randint(0, n - 1)

            row = rows[r_idx]
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else bound + 8
            upper = min(upper, bound + 15)
            if lower > upper:
                continue

            new_val = random.randint(lower, upper)
            if new_val == row[i_idx]:
                continue

            removed, _ = apply_change(rows, row_diffs, diff_counts, collision_count, r_idx, i_idx, new_val)
            if removed is None:
                continue

            new_max_scope = max(row[-1] for row in rows)
            new_cost = cost_of(collision_count, new_max_scope)
            delta = new_cost - cur_cost

            accept = False
            if delta <= 0:
                accept = True
            else:
                # Only accept worsening if it doesn't add collisions (delta < BIG)
                if delta < BIG:
                    p = math.exp(-delta / T)
                    accept = random.random() < p

            if accept:
                cur_cost = new_cost
                max_scope = new_max_scope
                step += 1
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_rows = [r[:] for r in rows]
                    best_max = max_scope
                    last_improve = time.time()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                collision_count = revert_change(rows, row_diffs, diff_counts, collision_count,
                                                r_idx, i_idx, row[i_idx], removed[0], removed[1])
                # note: after revert, row[i_idx] is old_val again

            T *= cooling
            if T < 0.05:
                T = 0.05

            # Diversification: if stuck for a long time, restart from best with perturbation
            if no_improve_steps > 4000 and (time.time() - last_improve) > 4.0:
                perturbed = [r[:] for r in best_rows]
                for _ in range(random.randint(2, 5)):
                    ri = random.randint(0, n - 1)
                    ii = random.randint(1, L - 1)
                    rr = perturbed[ri]
                    lo = rr[ii - 1] + 1
                    hi = rr[ii + 1] - 1 if ii < L - 1 else best_max + 6
                    if lo <= hi:
                        rr[ii] = random.randint(lo, hi)
                rows, row_diffs, diff_counts, collision_count, max_scope = make_state(perturbed)
                cur_cost = cost_of(collision_count, max_scope)
                T = 30.0
                no_improve_steps = 0
                last_improve = time.time()

        return best_rows, best_max, best_cost

    # ---------- High-precision final polish (exact local solve) ----------
    # Only accept collision-free moves that strictly reduce scope. Repeat until stable.
    def polish(rows_in, time_limit):
        rows, row_diffs, diff_counts, collision_count, max_scope = make_state(rows_in)
        if collision_count > 0:
            return rows_in, max(max(r[-1] for r in rows_in))
        best_max = max_scope
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            for r_idx in range(n):
                row = rows[r_idx]
                for i_idx in range(L - 1, 0, -1):  # last element first (biggest scope lever)
                    cur = row[i_idx]
                    lower = row[i_idx - 1] + 1
                    # Try to move down (reduce scope)
                    for new_val in range(cur - 1, lower - 1, -1):
                        if new_val >= cur:
                            continue
                        removed, _ = apply_change(rows, row_diffs, diff_counts, collision_count,
                                                  r_idx, i_idx, new_val)
                        if removed is None:
                            continue
                        if collision_count == 0:
                            new_max_scope = max(row[-1] for row in rows)
                            if new_max_scope < best_max:
                                best_max = new_max_scope
                                max_scope = new_max_scope
                                improved = True
                            # keep the change (it's collision-free and <= scope)
                        else:
                            collision_count = revert_change(rows, row_diffs, diff_counts,
                                                            collision_count, r_idx, i_idx,
                                                            cur, removed[0], removed[1])
            # early stop if time is tight
            if time.time() - start > time_limit - 1.0:
                break
        return rows, best_max

    # ---------- Main loop ----------
    while time.time() < deadline - 6.0:
        # 1) SA polish on the current best
        sa_time = min(12.0, deadline - time.time() - 6.0)
        if sa_time <= 1.0:
            break
        local_rows, local_max, local_cost = sa_optimize([r[:] for r in best_rows], sa_time, best_max)
        if local_cost < BIG:  # valid (no collisions)
            if local_max < best_max:
                best_rows = local_rows
                best_max = local_max

        # 2) High-precision final polish (cheap, exact)
        pol_time = min(4.0, deadline - time.time() - 5.0)
        if pol_time <= 0.3:
            break
        pol_rows, pol_max = polish(best_rows, pol_time)
        if pol_max < best_max:
            best_rows = pol_rows
            best_max = pol_max

    # Final safety: ensure we return a valid (collision-free) set if possible
    # best_rows from SA/polish may carry collisions; fall back to greedy if so.
    def is_valid(rows):
        seen = set()
        for r in rows:
            for d in get_diffs(r):
                if d in seen:
                    return False
                seen.add(d)
        return True

    if not is_valid(best_rows):
        # try a quick polish-free fix: keep best only if valid, else greedy
        if is_valid(greedy_construct()):
            pass
        # recompute validity of best_rows; if invalid, return greedy (always valid)
        best_rows = greedy_construct()
        best_max = max(r[-1] for r in best_rows)

    return best_rows
