import time
import random
import math
import bisect

MAX_SCOPE_ESTIMATE = 512  # bitmask covers differences below this

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    best_scope = float('inf')
    best_rows = None

    # Phase 1: Greedy initial solutions + iterative deepening restarts
    seed = 0
    scope_cap = MAX_SCOPE_ESTIMATE
    while time.time() < deadline - 70:
        rows = _greedy_construct(n_rows, k, seed, scope_cap)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lower_bound:
                    break
        # Iterative deepening: tighten the scope cap to force tighter packing
        if best_scope < float('inf'):
            scope_cap = max(lower_bound, best_scope - 1)
        else:
            scope_cap = max(120, scope_cap - 10)
        seed += 1

    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0, MAX_SCOPE_ESTIMATE)
        best_scope = _scope(best_rows)

    # Phase 2: Simulated annealing (bitmask conflict checks + soft-max surrogate)
    if time.time() < deadline - 15:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Final polish (per-element downward reduction + row swaps)
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs(row):
    n = len(row)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _diffs_set(row):
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _packing_score(new_diffs, used, x):
    if not used:
        return x * 1.0
    if not isinstance(used, list):
        used_sorted = sorted(used)
    else:
        used_sorted = used
    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist
    return total_dist / max(len(new_diffs), 1) + 0.1 * x

def _greedy_construct(n_rows, k, seed, scope_cap):
    rng = random.Random(seed)
    used = set()
    used_sorted = []
    rows = []

    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            scored_xs = []
            last_x = row[-1]
            limit = min(scope_cap, 500)

            x = last_x + 1
            while len(valid_xs) < 40 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1

            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 40 and x <= scope_cap:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        score = _packing_score(new_diffs, used_sorted, x)
                        scored_xs.append((score, x))
                    x += 1

            if not valid_xs:
                return None

            scored_xs.sort(key=lambda t: t[0])
            top_n = min(5, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(used_sorted, d)

        rows.append(row)

    return rows

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]

    # Precompute per-row diffs and global used set + bitmask
    row_diffs = [_diffs_set(r) for r in rows]
    global_used = set()
    for d in row_diffs:
        global_used.update(d)
    global_mask = 0
    for d in global_used:
        if d < MAX_SCOPE_ESTIMATE:
            global_mask |= (1 << d)

    T_start = 30.0
    T_end = 0.5
    total_steps = 30000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    T = T_start
    steps_since_improve = 0
    max_no_improve = 5000

    for step in range(total_steps):
        if time.time() > deadline - 5:
            break

        if rng.random() < 0.3:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)

        old_row = list(rows[row_idx])
        old_diffs = row_diffs[row_idx]

        # Candidate row avoiding global_used (which includes this row's diffs)
        new_row = _generate_candidate_row(rows, row_idx, k, rng, global_used, global_mask)
        if new_row is None:
            continue

        new_diffs = _diffs_set(new_row)
        # other_diffs = global_used minus this row's old diffs
        other_diffs = global_used - old_diffs

        if new_diffs & other_diffs:
            continue

        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)

        # Soft-max surrogate: scope + small soft-max of per-row last elements
        soft = 0.0
        for r in new_rows:
            soft += r[-1]
        surrogate = new_scope + 0.05 * soft / n_rows

        delta = surrogate - (current_scope + 0.05 * sum(r[-1] for r in rows) / n_rows)
        if delta <= 0:
            rows = new_rows
            row_diffs[row_idx] = new_diffs
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                rows = new_rows
                row_diffs[row_idx] = new_diffs
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1

        T *= cool_rate

        if steps_since_improve > max_no_improve:
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row(rows, pert_idx, k, rng, global_used, global_mask)
            if pert_row is not None:
                pert_diffs = _diffs_set(pert_row)
                other = global_used - row_diffs[pert_idx]
                if not (pert_diffs & other):
                    rows[pert_idx] = pert_row
                    row_diffs[pert_idx] = pert_diffs
                    current_scope = _scope(rows)
            steps_since_improve = 0

    return best_rows

def _generate_candidate_row(rows, exclude_idx, k, rng, used_diffs, used_mask):
    used_sorted = sorted(used_diffs)
    row = [0]
    local_used = set(used_diffs)
    local_mask = used_mask
    local_sorted = list(used_sorted)

    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 500

        x = last_x + 1
        while len(valid_xs) < 30 and x <= limit:
            # Fast bitmask conflict check
            conflict = False
            for a in row:
                d = x - a
                if d < MAX_SCOPE_ESTIMATE:
                    if local_mask & (1 << d):
                        conflict = True
                        break
                elif d in local_used:
                    conflict = True
                    break
            if not conflict:
                new_diffs = {x - a for a in row}
                valid_xs.append(x)
                score = _packing_score(new_diffs, local_sorted, x)
                scored_xs.append((score, x))
            x += 1

        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 30:
                conflict = False
                for a in row:
                    d = x - a
                    if d < MAX_SCOPE_ESTIMATE:
                        if local_mask & (1 << d):
                            conflict = True
                            break
                    elif d in local_used:
                        conflict = True
                        break
                if not conflict:
                    new_diffs = {x - a for a in row}
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, local_sorted, x)
                    scored_xs.append((score, x))
                x += 1

        if not valid_xs:
            return None

        scored_xs.sort(key=lambda t: t[0])
        top_n = min(4, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        local_used.update(new_diffs)
        for d in new_diffs:
            if d < MAX_SCOPE_ESTIMATE:
                local_mask |= (1 << d)
            bisect.insort(local_sorted, d)

    return row

def _polish_max(rows, n_rows, k, start_time, deadline):
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    max_iter = 200

    row_diffs = [_diffs_set(r) for r in rows]
    global_used = set()
    for d in row_diffs:
        global_used.update(d)
    global_mask = 0
    for d in global_used:
        if d < MAX_SCOPE_ESTIMATE:
            global_mask |= (1 << d)

    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break

        improved = False

        # Pass A: per-element downward reduction on the max row
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        row = rows[max_idx]
        for pos in range(k, 0, -1):
            old_val = row[pos]
            for new_val in range(row[pos - 1] + 1, old_val):
                # Diffs removed and added involving this element
                removed = set()
                added = set()
                for i in range(len(row)):
                    if i != pos:
                        removed.add(abs(old_val - row[i]))
                        added.add(abs(new_val - row[i]))
                # other = global_used minus this row's diffs, minus removed
                other = global_used - row_diffs[max_idx] - removed
                if added & other:
                    continue
                # valid move
                row[pos] = new_val
                row_diffs[max_idx] = _diffs_set(row)
                global_used = (global_used - removed) | added
                global_mask = 0
                for d in global_used:
                    if d < MAX_SCOPE_ESTIMATE:
                        global_mask |= (1 << d)
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                improved = True
                break
            if improved:
                break

        # Pass B: try swapping elements between two rows
        if not improved:
            for r1 in range(n_rows):
                for r2 in range(r1 + 1, n_rows):
                    for p1 in range(1, len(rows[r1])):
                        for p2 in range(1, len(rows[r2])):
                            v1 = rows[r1][p1]
                            v2 = rows[r2][p2]
                            if v1 == v2:
                                continue
                            # Tentative swap
                            rows[r1][p1] = v2
                            rows[r2][p2] = v1
                            # Check validity
                            valid = True
                            used = set()
                            for rr in rows:
                                for i in range(len(rr)):
                                    for j in range(i + 1, len(rr)):
                                        d = rr[j] - rr[i]
                                        if d in used:
                                            valid = False
                                            break
                                        used.add(d)
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if not valid:
                                rows[r1][p1] = v1
                                rows[r2][p2] = v2
                                continue
                            new_scope = _scope(rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [list(r) for r in rows]
                                row_diffs = [_diffs_set(r) for r in rows]
                                global_used = set()
                                for d in row_diffs:
                                    global_used.update(d)
                                global_mask = 0
                                for d in global_used:
                                    if d < MAX_SCOPE_ESTIMATE:
                                        global_mask |= (1 << d)
                                improved = True
                                break
                            else:
                                rows[r1][p1] = v1
                                rows[r2][p2] = v2
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    if best_scope < _scope([list(r) for r in rows]):
        return best_rows
    return None
