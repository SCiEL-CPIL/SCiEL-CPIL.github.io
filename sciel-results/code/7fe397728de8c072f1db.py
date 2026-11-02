import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _row_diffs_set(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs

def _compute_all_used(rows):
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    return max(max(row) for row in rows)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112.0

    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)

    # 2. Simulated Annealing (fast single-pass inner loop)
    current_rows = [r[:] for r in rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]

    T = 250.0
    T_min = 0.05
    cool = 0.99999
    no_improve = 0
    max_no_improve = 40000
    steps = 0
    n_pos = k  # positions 1..k

    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position to modify
        # Bias toward rows that have the max entry
        if random.random() < 0.7 and max_row_indices:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)

        old_val = current_rows[ri][pi]
        old_diffs = row_diffs[ri]
        old_row_max = row_maxes[ri]

        # Generate candidate values with bias toward reduction
        min_allowed = current_rows[ri][pi - 1] + 1
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40

        # Try a handful of candidate moves
        n_cand = random.randint(4, 10)
        # Decreasing candidates
        max_decrease = old_val - min_allowed
        if max_decrease > 0:
            # Sample from decreasing range, biased to larger decreases
            for _ in range(n_cand):
                d = random.randint(1, max_decrease)
                # Bias toward larger d
                d = max(1, d + d // 2)
                if d > max_decrease:
                    d = max_decrease
                candidates_val = old_val - d
                if candidates_val <= min_allowed:
                    continue
                if pi < k and candidates_val >= current_rows[ri][pi + 1]:
                    continue
                # Build candidate row
                cand_row = current_rows[ri][:]
                cand_row[pi] = candidates_val
                new_val = candidates_val

                # Compute new diffs involving pi
                new_only_diffs = []
                valid_internal = True
                for i in range(len(cand_row)):
                    if i == pi:
                        continue
                    d = abs(new_val - cand_row[i])
                    if d in old_diffs:
                        # This diff already exists in this row - check if it's from another pair
                        # Actually we need to check against unchanged diffs
                        pass
                    new_only_diffs.append(d)

                # Check internal validity: new_only_diffs must not collide with unchanged diffs
                # Unchanged diffs = old_diffs minus diffs that involved pi in the old row
                # Build set of old diffs involving pi
                old_pi_diffs = set()
                for i in range(len(old_val and current_rows[ri])):
                    if i == pi:
                        continue
                    old_pi_diffs.add(abs(old_val - current_rows[ri][i]))
                unchanged_diffs = old_diffs - old_pi_diffs

                conflict_internal = False
                conflict_global = False
                for d in new_only_diffs:
                    if d in unchanged_diffs:
                        conflict_internal = True
                        break
                    if d in all_used and d not in old_diffs:
                        conflict_global = True
                        break
                if conflict_internal or conflict_global:
                    continue

                # Build new_diffs set
                new_diffs = unchanged_diffs | set(new_only_diffs)

                # Compute new row max
                if pi == k:
                    cand_row_max = new_val
                else:
                    cand_row_max = max(old_row_max, new_val) if new_val > old_row_max else old_row_max
                    # Actually need to be careful
                    cand_row_max = max(cand_row)

                # New global max
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max

                delta = new_global_max - current_val

                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.5:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True

                if accept:
                    all_used -= old_diffs
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)

                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                        max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                    else:
                        no_improve += 1
                    break
            else:
                no_improve += 1
        else:
            no_improve += 1

        steps += 1
        T = max(T_min, T * cool)

        # Periodic restart if stuck
        if no_improve > 18000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
            T = 250.0
            no_improve = 0

    # 3. Final targeted polish: focus on reducing the max
    polish_deadline = min(deadline, start_time + 116.0)
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    no_improve = 0

    while time.time() < polish_deadline and no_improve < 30000:
        if max_row_indices and random.random() < 0.85:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        # Focus on the last position of max rows
        if ri in max_row_indices and random.random() < 0.6:
            pi = k
        else:
            pi = random.randrange(1, k + 1)

        old_val = current_rows[ri][pi]
        old_diffs = row_diffs[ri]
        old_row_max = row_maxes[ri]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = old_val - min_allowed
        if max_decrease <= 0:
            no_improve += 1
            continue

        # Try several decreasing candidates
        n_try = min(max_decrease, random.randint(6, 14))
        for _ in range(n_try):
            d = random.randint(1, max_decrease)
            d = max(1, d + d // 3)  # bias to larger
            if d > max_decrease:
                d = max_decrease
            new_val = old_val - d
            if new_val <= min_allowed:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue

            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val

            new_only_diffs = [abs(new_val - cand_row[i]) for i in range(len(cand_row)) if i != pi]

            old_pi_diffs = set()
            for i in range(len(current_rows[ri])):
                if i == pi:
                    continue
                old_pi_diffs.add(abs(old_val - current_rows[ri][i]))
            unchanged_diffs = old_diffs - old_pi_diffs

            conflict = False
            for d in new_only_diffs:
                if d in unchanged_diffs:
                    conflict = True
                    break
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            if conflict:
                continue

            new_diffs = unchanged_diffs | set(new_only_diffs)
            cand_row_max = max(cand_row)

            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max

            delta = new_global_max - current_val
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.25
            else:
                accept = False

            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)

                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                break
        else:
            no_improve += 1

    # 4. Very final micro-polish
    micro_deadline = start_time + 118.0
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]

    while time.time() < micro_deadline:
        # Try to reduce the global max by 1
        target = current_val
        found = False
        for ri in range(n_rows):
            if row_maxes[ri] != target:
                continue
            pi = k  # last position
            old_val = current_rows[ri][pi]
            if old_val != target:
                continue
            old_diffs = row_diffs[ri]
            min_allowed = current_rows[ri][pi - 1] + 1
            new_val = target - 1
            if new_val < min_allowed:
                continue
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            new_only_diffs = [abs(new_val - cand_row[i]) for i in range(len(cand_row)) if i != pi]
            old_pi_diffs = set()
            for i in range(len(current_rows[ri])):
                if i == pi:
                    continue
                old_pi_diffs.add(abs(old_val - current_rows[ri][i]))
            unchanged_diffs = old_diffs - old_pi_diffs
            conflict = False
            for d in new_only_diffs:
                if d in unchanged_diffs or (d in all_used and d not in old_diffs):
                    conflict = True
                    break
            if conflict:
                continue
            new_diffs = unchanged_diffs | set(new_only_diffs)
            all_used -= old_diffs
            all_used |= new_diffs
            current_rows[ri] = cand_row
            row_diffs[ri] = new_diffs
            row_maxes[ri] = new_val
            current_val = max(row_maxes)
            best_val = current_val
            best_rows = [r[:] for r in current_rows]
            found = True
            break
        if not found:
            # Try random small moves
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            old_diffs = row_diffs[ri]
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = old_val - min_allowed
            if max_decrease <= 0:
                continue
            d = random.randint(1, min(max_decrease, 15))
            new_val = old_val - d
            if new_val <= min_allowed:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            new_only_diffs = [abs(new_val - cand_row[i]) for i in range(len(cand_row)) if i != pi]
            old_pi_diffs = set()
            for i in range(len(current_rows[ri])):
                if i == pi:
                    continue
                old_pi_diffs.add(abs(old_val - current_rows[ri][i]))
            unchanged_diffs = old_diffs - old_pi_diffs
            conflict = False
            for d in new_only_diffs:
                if d in unchanged_diffs or (d in all_used and d not in old_diffs):
                    conflict = True
                    break
            if conflict:
                continue
            new_diffs = unchanged_diffs | set(new_only_diffs)
            cand_row_max = max(cand_row)
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = max(current_rows[ri])
            if new_global_max <= current_val:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]

    # Validate and return
    return best_rows

# EVOLVE-BLOCK-END
