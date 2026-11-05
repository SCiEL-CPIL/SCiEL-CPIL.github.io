import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    n_marks = k + 1
    diffs_per_row = k * (k + 1) // 2

    def _time_left():
        return max(0.0, deadline - time.time())

    def get_row_diffs(row):
        diffs = set()
        for i in range(n_marks):
            for j in range(i + 1, n_marks):
                diffs.add(row[j] - row[i])
        return diffs

    def compute_cost(rows, row_diffs=None):
        """Compute cost: scope + conflicts * 100"""
        used = set()
        max_entry = 0
        conflicts = 0
        for r in range(n_rows):
            row = rows[r]
            max_entry = max(max_entry, row[-1])
            if row_diffs is not None:
                rd = row_diffs[r]
            else:
                rd = get_row_diffs(row)
            for d in rd:
                if d in used:
                    conflicts += 1
                else:
                    used.add(d)
        return max_entry + conflicts * 100

    def greedy_row(used, k, strategy="pure", rng=None):
        """Build one row greedily with various strategies."""
        row = [0]
        x = 0
        while len(row) < n_marks:
            if strategy == "pure":
                x += 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used |= {x - a for a in row[:-1]}
            elif strategy == "shuffled":
                base = x + 1
                window = 25
                candidates = list(range(base, base + window))
                if rng:
                    rng.shuffle(candidates)
                found = False
                for cand in candidates:
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        row.append(cand)
                        used |= {cand - a for a in row[:-1]}
                        x = cand
                        found = True
                        break
                if not found:
                    cand = base + window
                    while True:
                        new_diffs = {cand - a for a in row}
                        if not new_diffs & used:
                            row.append(cand)
                            used |= {cand - a for a in row[:-1]}
                            x = cand
                            break
            elif strategy == "pressure":
                # Difference pressure: choose candidate that minimizes max used diff
                base = x + 1
                best_cand = None
                best_max_diff = float('inf')
                for cand in range(base, base + 30):
                    new_diffs = {cand - a for a in row}
                    if new_diffs & used:
                        continue
                    # Score: minimize the maximum new difference
                    max_new = max(new_diffs)
                    if max_new < best_max_diff:
                        best_max_diff = max_new
                        best_cand = cand
                if best_cand is None:
                    cand = base
                    while True:
                        new_diffs = {cand - a for a in row}
                        if not new_diffs & used:
                            best_cand = cand
                            break
                        cand += 1
                row.append(best_cand)
                used |= {best_cand - a for a in row[:-1]}
                x = best_cand
            elif strategy == "lookahead":
                base = x + 1
                valid_cands = []
                for cand in range(base, base + 35):
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        valid_cands.append(cand)
                    if len(valid_cands) >= 5:
                        break
                if not valid_cands:
                    cand = base
                    while True:
                        new_diffs = {cand - a for a in row}
                        if not new_diffs & used:
                            valid_cands.append(cand)
                            break
                        cand += 1
                # Pick candidate with most future options
                best_cand = valid_cands[0]
                best_score = -1
                for cand in valid_cands[:3]:
                    new_diffs = {cand - a for a in row}
                    future = 0
                    for nc in range(cand + 1, cand + 15):
                        nd = {nc - a for a in row} | {nc - cand}
                        if not (nd & (used | new_diffs)):
                            future += 1
                    score = future * 10 - cand
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                row.append(best_cand)
                used |= {best_cand - a for a in row[:-1]}
                x = best_cand
        return row

    def backtrack_row(used, k, max_scope, rng):
        """Build one row with all entries <= max_scope using backtracking."""
        row = [0]
        
        def bt():
            if len(row) == n_marks:
                return True
            start = row[-1] + 1
            end = min(max_scope + 1, start + 40)
            candidates = list(range(start, end))
            if rng and len(candidates) > 6:
                rng.shuffle(candidates[3:])
            for x in candidates:
                if x > max_scope:
                    break
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                row.append(x)
                used.update(new_diffs)
                if bt():
                    return True
                row.pop()
                used.difference_update(new_diffs)
            return False
        
        if bt():
            return row[:]
        return None

    def all_diffs_except(rows, row_diffs, skip_idx):
        """Get set of all differences except those in rows[skip_idx]."""
        s = set()
        for idx in range(n_rows):
            if idx == skip_idx:
                continue
            s |= row_diffs[idx]
        return s

    def simulated_annealing(rows, time_budget):
        current_rows = [list(r) for r in rows]
        row_diffs = [get_row_diffs(r) for r in current_rows]
        current_cost = compute_cost(current_rows, row_diffs)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost
        best_scope = max(r[-1] for r in best_rows)

        T_start = 60.0
        T_end = 0.05
        max_steps = 200000
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_steps - 1))
        rng = random.Random(hash(tuple(tuple(r) for r in rows)) % (2**32))

        steps = 0
        while _time_left() > 2 and steps < max_steps:
            steps += 1
            T = T_start * (cooling_rate ** steps)
            if T < T_end:
                T = T_end

            r = rng.randint(0, n_rows - 1)
            move_type = rng.random()

            if move_type < 0.35:
                # Rebuild row with backtracking at best_scope
                used_other = all_diffs_except(current_rows, row_diffs, r)
                new_row = backtrack_row(used_other, k, best_scope, rng)
                if new_row is not None:
                    old_row = current_rows[r]
                    old_rd = row_diffs[r]
                    current_rows[r] = new_row
                    new_rd = get_row_diffs(new_row)
                    new_cost = compute_cost(current_rows, row_diffs)
                    # Recompute with updated row_diffs
                    row_diffs[r] = new_rd
                    new_cost = compute_cost(current_rows, row_diffs)
                    delta = new_cost - current_cost
                    if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_rows = [list(row) for row in current_rows]
                            new_scope = max(row[-1] for row in best_rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                    else:
                        current_rows[r] = old_row
                        row_diffs[r] = old_rd
                continue

            elif move_type < 0.55:
                # Rebuild row with slightly larger scope (escape)
                escape_scope = best_scope + rng.randint(1, 8)
                used_other = all_diffs_except(current_rows, row_diffs, r)
                new_row = backtrack_row(used_other, k, escape_scope, rng)
                if new_row is not None:
                    old_row = current_rows[r]
                    old_rd = row_diffs[r]
                    current_rows[r] = new_row
                    new_rd = get_row_diffs(new_row)
                    row_diffs[r] = new_rd
                    new_cost = compute_cost(current_rows, row_diffs)
                    delta = new_cost - current_cost
                    if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_rows = [list(row) for row in current_rows]
                    else:
                        current_rows[r] = old_row
                        row_diffs[r] = old_rd
                continue

            else:
                # Perturb single element
                pos = rng.randint(1, k)
                old_val = current_rows[r][pos]
                min_val = current_rows[r][pos - 1] + 1
                max_val = current_rows[r][pos + 1] - 1 if pos < k else best_scope + 20

                if min_val > max_val:
                    continue

                if rng.random() < 0.4:
                    new_val = rng.randint(min_val, min_val + max(1, (max_val - min_val) // 3))
                elif rng.random() < 0.6:
                    delta_v = rng.randint(-4, 4)
                    new_val = old_val + delta_v
                else:
                    new_val = rng.randint(min_val, max_val)

                if new_val < min_val or new_val > max_val or new_val == old_val:
                    continue

                old_row = current_rows[r]
                old_rd = row_diffs[r]
                new_row = old_row[:]
                new_row[pos] = new_val
                new_rd = get_row_diffs(new_row)

                current_rows[r] = new_row
                row_diffs[r] = new_rd
                new_cost = compute_cost(current_rows, row_diffs)
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows = [list(row) for row in current_rows]
                        new_scope = max(row[-1] for row in best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                else:
                    current_rows[r] = old_row
                    row_diffs[r] = old_rd

        return best_rows, best_cost

    def local_search(rows, time_budget):
        current_rows = [list(r) for r in rows]
        row_diffs = [get_row_diffs(r) for r in current_rows]
        current_cost = compute_cost(current_rows, row_diffs)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost

        no_improve = 0
        max_no_improve = 5000
        rng = random.Random(42)

        while _time_left() > 3 and no_improve < max_no_improve:
            improved = False
            for r in range(n_rows):
                for pos in range(1, k + 1):
                    min_val = current_rows[r][pos - 1] + 1
                    max_val = current_rows[r][pos + 1] - 1 if pos < k else max(current_rows[r][-1] + 15, 60)
                    if min_val > max_val:
                        continue

                    best_local_val = current_rows[r][pos]
                    best_local_cost = current_cost
                    old_rd = row_diffs[r]

                    candidates = set()
                    candidates.add(current_rows[r][pos])
                    candidates.add(min_val)
                    if current_rows[r][pos] > min_val:
                        candidates.add(current_rows[r][pos] - 1)
                        candidates.add(current_rows[r][pos] - 2)
                    if current_rows[r][pos] < max_val:
                        candidates.add(current_rows[r][pos] + 1)
                    for _ in range(3):
                        candidates.add(rng.randint(min_val, min_val + max(1, (max_val - min_val) // 4)))

                    for val in candidates:
                        if val < min_val or val > max_val:
                            continue
                        current_rows[r][pos] = val
                        new_rd = get_row_diffs(current_rows[r])
                        row_diffs[r] = new_rd
                        new_cost = compute_cost(current_rows, row_diffs)
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_val = val

                    current_rows[r][pos] = best_local_val
                    row_diffs[r] = get_row_diffs(current_rows[r])
                    if best_local_cost < current_cost:
                        current_cost = best_local_cost
                        improved = True

            if not improved:
                no_improve += 1
            else:
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [list(row) for row in current_rows]

        return best_rows, best_cost

    def final_polish(rows, time_budget):
        current_rows = [list(r) for r in rows]
        row_diffs = [get_row_diffs(r) for r in current_rows]
        current_cost = compute_cost(current_rows, row_diffs)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost

        for _ in range(100):
            if _time_left() < 2:
                break
            max_r = max(range(n_rows), key=lambda r: current_rows[r][-1])
            max_val = current_rows[max_r][-1]
            if max_val <= 0:
                break

            # Try shifting the last element down
            for shift in range(1, 5):
                new_last = max_val - shift
                if new_last <= current_rows[max_r][n_marks - 2]:
                    break
                old_row = current_rows[max_r]
                current_rows[max_r][n_marks - 1] = new_last
                new_rd = get_row_diffs(current_rows[max_r])
                old_rd = row_diffs[max_r]
                row_diffs[max_r] = new_rd
                new_cost = compute_cost(current_rows, row_diffs)
                if new_cost <= current_cost:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows = [list(row) for row in current_rows]
                else:
                    current_rows[max_r] = old_row
                    row_diffs[max_r] = old_rd
                    break

        return best_rows, best_cost

    # Multi-start greedy with different strategies
    strategies = ["pressure", "shuffled", "lookahead", "pure"]
    best_overall_rows = None
    best_overall_cost = float('inf')

    for strategy in strategies:
        if _time_left() < 10:
            break
        used = set()
        rows = []
        for r in range(n_rows):
            row = greedy_row(used, k, strategy=strategy)
            rows.append(row)
            used |= get_row_diffs(row)

        # Quick SA refinement
        sa_budget = min(15, _time_left() * 0.25)
        sa_rows, sa_cost = simulated_annealing(rows, sa_budget)

        # Quick local search
        ls_budget = min(10, _time_left() * 0.2)
        ls_rows, ls_cost = local_search(sa_rows, ls_budget)

        if ls_cost < best_overall_cost:
            best_overall_cost = ls_cost
            best_overall_rows = ls_rows

    # Final polish on the best
    polish_budget = min(10, _time_left() * 0.15)
    polished_rows, polished_cost = final_polish(best_overall_rows, polish_budget)
    if polished_cost < best_overall_cost:
        best_overall_cost = polished_cost
        best_overall_rows = polished_rows

    # If time permits, do one more round of SA from the polished result
    if _time_left() > 20:
        sa2_budget = min(20, _time_left() * 0.4)
        sa2_rows, sa2_cost = simulated_annealing(best_overall_rows, sa2_budget)
        if sa2_cost < best_overall_cost:
            best_overall_cost = sa2_cost
            best_overall_rows = sa2_rows
        if _time_left() > 10:
            ls2_budget = min(10, _time_left() * 0.3)
            ls2_rows, ls2_cost = local_search(best_overall_rows, ls2_budget)
            if ls2_cost < best_overall_cost:
                best_overall_cost = ls2_cost
                best_overall_rows = ls2_rows

    # Final validation and cleanup
    for r in range(n_rows):
        for m in range(1, n_marks):
            if best_overall_rows[r][m] <= best_overall_rows[r][m - 1]:
                best_overall_rows[r][m] = best_overall_rows[r][m - 1] + 1
        for m in range(n_marks):
            if best_overall_rows[r][m] < 0:
                best_overall_rows[r][m] = 0
            if m > 0 and best_overall_rows[r][m] <= best_overall_rows[r][m - 1]:
                best_overall_rows[r][m] = best_overall_rows[r][m - 1] + 1

    return best_overall_rows
