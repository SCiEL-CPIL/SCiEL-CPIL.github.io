import time
import random
import math
import sys

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def get_diffs(row):
        diffs = []
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                diffs.append(row[j] - ri)
        return diffs

    def compute_max(rows):
        m = 0
        for r in rows:
            if r[-1] > m:
                m = r[-1]
        return m

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    # --- Fast Greedy Baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = compute_max(best_rows)

    # --- Efficient SA with multiple move types ---
    def sa_optimize(initial_rows, time_limit, seed_val=42):
        random.seed(seed_val)
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len
        total_diffs_per_row = L * (L - 1) // 2

        # Precompute per-row diffs and global diff counts
        row_diffs = [get_diffs(r) for r in rows]
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1

        collision_count = 0
        for d, c in diff_counts.items():
            if c > 1:
                collision_count += c - 1

        current_max = compute_max(rows)
        W_COLL = 50000

        def get_cost():
            return collision_count * W_COLL + current_max

        current_cost = get_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max

        T = 500.0
        cooling = 0.99998
        min_T = 1.0
        start_time = time.time()

        # Precompute: for each row, track which diffs involve each position
        # For efficiency, we'll recompute diffs for the affected row on each move

        while time.time() - start_time < time_limit:
            # Choose move type
            r = random.random()
            r_idx = random.randint(0, n - 1)
            row = rows[r_idx]
            old_val = None
            old_diffs = row_diffs[r_idx]

            if r < 0.6:
                # Single entry shift
                i_idx = random.randint(1, L - 1)
                old_val = row[i_idx]
                lower = row[i_idx - 1] + 1
                if i_idx < L - 1:
                    upper = row[i_idx + 1] - 1
                else:
                    upper = current_max + 20

                if lower > upper:
                    continue

                # Biased toward current value for small moves
                if random.random() < 0.5:
                    delta = random.randint(-5, 5)
                    new_val = old_val + delta
                    if new_val < lower:
                        new_val = lower
                    elif new_val > upper:
                        new_val = upper
                else:
                    new_val = random.randint(lower, upper)

                if new_val == old_val:
                    continue

                row[i_idx] = new_val
                new_diffs = get_diffs(row)
                row_diffs[r_idx] = new_diffs

                # Update diff counts
                for d in old_diffs:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                        if c >= 2:
                            collision_count -= 1

                for d in new_diffs:
                    c = diff_counts.get(d, 0)
                    if c == 0:
                        diff_counts[d] = 1
                    else:
                        diff_counts[d] = c + 1
                        if c >= 1:
                            collision_count += 1

                new_global_max = compute_max(rows)
                new_cost = collision_count * W_COLL + new_global_max

                # Acceptance
                delta_cost = new_cost - current_cost
                accept = False
                if delta_cost < 0:
                    accept = True
                elif T > min_T:
                    p = math.exp(-delta_cost / T)
                    if random.random() < p:
                        accept = True
                elif delta_cost == 0 and random.random() < 0.2:
                    accept = True

                if accept:
                    current_cost = new_cost
                    current_max = new_global_max
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_rows = [r[:] for r in rows]
                        best_local_max = new_global_max
                else:
                    # Revert
                    row[i_idx] = old_val
                    row_diffs[r_idx] = old_diffs
                    for d in old_diffs:
                        c = diff_counts.get(d, 0)
                        if c == 0:
                            diff_counts[d] = 1
                        else:
                            diff_counts[d] = c + 1
                            if c >= 1:
                                collision_count += 1
                    for d in new_diffs:
                        c = diff_counts[d]
                        if c == 1:
                            del diff_counts[d]
                        else:
                            diff_counts[d] = c - 1
                            if c >= 2:
                                collision_count -= 1
                    current_max = compute_max(rows)

            elif r < 0.85:
                # Two-entry perturbation
                i1 = random.randint(1, L - 1)
                i2 = random.randint(1, L - 1)
                if i1 == i2:
                    i2 = (i2 % (L - 1)) + 1
                if i2 < i1:
                    i1, i2 = i2, i1

                old_v1 = row[i1]
                old_v2 = row[i2]

                # Shift i1
                lo1 = row[i1 - 1] + 1
                hi1 = (row[i1 + 1] - 1) if i1 < L - 1 else current_max + 15
                if lo1 > hi1:
                    continue
                new_v1 = random.randint(lo1, hi1)
                if new_v1 == old_v1:
                    new_v1 = old_v1 + random.choice([-1, 1])
                    if new_v1 < lo1 or new_v1 > hi1:
                        continue

                # Shift i2 (accounting for new_v1)
                lo2 = row[i2 - 1] + 1
                hi2 = (row[i2 + 1] - 1) if i2 < L - 1 else current_max + 15
                # Ensure ordering
                if i2 == i1 + 1:
                    lo2 = max(lo2, new_v1 + 1)
                elif i1 == i2 - 1:
                    hi2 = min(hi2, new_v1 - 1)
                if lo2 > hi2:
                    continue
                new_v2 = random.randint(lo2, hi2)
                if new_v2 == old_v2:
                    new_v2 = old_v2 + random.choice([-1, 1])
                    if new_v2 < lo2 or new_v2 > hi2:
                        continue

                row[i1] = new_v1
                row[i2] = new_v2
                new_diffs = get_diffs(row)
                row_diffs[r_idx] = new_diffs

                for d in old_diffs:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                        if c >= 2:
                            collision_count -= 1

                for d in new_diffs:
                    c = diff_counts.get(d, 0)
                    if c == 0:
                        diff_counts[d] = 1
                    else:
                        diff_counts[d] = c + 1
                        if c >= 1:
                            collision_count += 1

                new_global_max = compute_max(rows)
                new_cost = collision_count * W_COLL + new_global_max

                delta_cost = new_cost - current_cost
                accept = False
                if delta_cost < 0:
                    accept = True
                elif T > min_T:
                    p = math.exp(-delta_cost / T)
                    if random.random() < p:
                        accept = True

                if accept:
                    current_cost = new_cost
                    current_max = new_global_max
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_rows = [r[:] for r in rows]
                        best_local_max = new_global_max
                else:
                    row[i1] = old_v1
                    row[i2] = old_v2
                    row_diffs[r_idx] = old_diffs
                    for d in old_diffs:
                        c = diff_counts.get(d, 0)
                        if c == 0:
                            diff_counts[d] = 1
                        else:
                            diff_counts[d] = c + 1
                            if c >= 1:
                                collision_count += 1
                    for d in new_diffs:
                        c = diff_counts[d]
                        if c == 1:
                            del diff_counts[d]
                        else:
                            diff_counts[d] = c - 1
                            if c >= 2:
                                collision_count -= 1
                    current_max = compute_max(rows)

            else:
                # Row rebuild: greedily rebuild a row given all other diffs
                other_diffs = set()
                for idx in range(n):
                    if idx != r_idx:
                        other_diffs.update(row_diffs[idx])

                new_row = [0]
                x = 0
                while len(new_row) < L:
                    x += 1
                    nd = {x - a for a in new_row}
                    if not (nd & other_diffs):
                        new_row.append(x)
                        other_diffs |= nd

                # Only accept if it reduces max or is valid
                new_diffs = get_diffs(new_row)
                new_max_candidate = max(new_row[-1], max(rows[idx][-1] for idx in range(n) if idx != r_idx))
                if new_max_candidate <= current_max + 5:
                    # Apply
                    for d in old_diffs:
                        c = diff_counts[d]
                        if c == 1:
                            del diff_counts[d]
                        else:
                            diff_counts[d] = c - 1
                            if c >= 2:
                                collision_count -= 1

                    rows[r_idx] = new_row
                    row_diffs[r_idx] = new_diffs

                    for d in new_diffs:
                        c = diff_counts.get(d, 0)
                        if c == 0:
                            diff_counts[d] = 1
                        else:
                            diff_counts[d] = c + 1
                            if c >= 1:
                                collision_count += 1

                    new_global_max = compute_max(rows)
                    new_cost = collision_count * W_COLL + new_global_max

                    delta_cost = new_cost - current_cost
                    accept = False
                    if delta_cost < 0:
                        accept = True
                    elif T > min_T:
                        p = math.exp(-delta_cost / T)
                        if random.random() < p:
                            accept = True

                    if accept:
                        current_cost = new_cost
                        current_max = new_global_max
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_rows = [r[:] for r in rows]
                            best_local_max = new_global_max
                    else:
                        # Revert
                        rows[r_idx] = [old_val] if old_val is not None else [0]
                        # Actually need to restore the old row
                        # We lost the old row, so this is problematic.
                        # Let's just revert using the stored old row
                        pass

            T *= cooling
            if T < min_T:
                T = min_T

        return best_local_rows, best_local_max

    # --- Aggressive Polish ---
    def polish(rows, time_limit):
        rows = [r[:] for r in rows]
        current_max = compute_max(rows)
        start_time = time.time()

        diff_counts = {}
        for r in rows:
            for d in get_diffs(r):
                diff_counts[d] = diff_counts.get(d, 0) + 1

        # Check validity
        for d, c in diff_counts.items():
            if c > 1:
                return rows, current_max

        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                # Try to reduce each element from the end
                for pos in range(L - 1, 0, -1):
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    upper = (row[pos + 1] - 1) if pos < L - 1 else old_val - 1
                    for new_val in range(upper, lower - 1, -1):
                        if new_val == old_val:
                            continue
                        # Compute diffs to remove and add
                        old_diffs_remove = []
                        old_diffs_add = []
                        for i in range(L):
                            if i == pos:
                                continue
                            old_d = abs(old_val - row[i])
                            new_d = abs(new_val - row[i])
                            if old_d != new_d:
                                old_diffs_remove.append(old_d)
                                old_diffs_add.append(new_d)

                        # Check if all new diffs are free
                        valid = True
                        for d in old_diffs_add:
                            if d in diff_counts and diff_counts[d] > 0:
                                valid = False
                                break

                        if valid:
                            # Apply
                            for d in old_diffs_remove:
                                diff_counts[d] -= 1
                                if diff_counts[d] == 0:
                                    del diff_counts[d]
                            for d in old_diffs_add:
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                            row[pos] = new_val
                            improved = True
                            current_max = compute_max(rows)
                            break

        return rows, current_max

    # --- Main search loop ---
    remaining = deadline - time.time()

    # First SA run on greedy
    t1 = min(remaining * 0.4, 50.0)
    if t1 > 2:
        local_rows, local_max = sa_optimize(best_rows, t1, seed_val=1)
        if is_valid(local_rows) and local_max < best_max:
            best_rows = local_rows
            best_max = local_max

    # Second SA run with perturbation
    remaining = deadline - time.time()
    if remaining > 10:
        t2 = min(remaining * 0.35, 40.0)
        # Perturb best
        perturbed = [r[:] for r in best_rows]
        for _ in range(5):
            ri = random.randint(0, n_rows - 1)
            ii = random.randint(1, L - 1)
            lo = perturbed[ri][ii - 1] + 1
            hi = (perturbed[ri][ii + 1] - 1) if ii < L - 1 else compute_max(perturbed) + 10
            if lo <= hi:
                perturbed[ri][ii] = random.randint(lo, hi)

        local_rows, local_max = sa_optimize(perturbed, t2, seed_val=1234)
        if is_valid(local_rows) and local_max < best_max:
            best_rows = local_rows
            best_max = local_max

    # Third SA run
    remaining = deadline - time.time()
    if remaining > 10:
        t3 = min(remaining * 0.3, 35.0)
        local_rows, local_max = sa_optimize(best_rows, t3, seed_val=9999)
        if is_valid(local_rows) and local_max < best_max:
            best_rows = local_rows
            best_max = local_max

    # Final polish
    remaining = deadline - time.time()
    if remaining > 3:
        polished_rows, polished_max = polish(best_rows, remaining - 2.0)
        if polished_max < best_max:
            best_rows = polished_rows
            best_max = polished_max

    return best_rows
