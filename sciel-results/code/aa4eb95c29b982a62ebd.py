import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0

    k_plus_1 = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    lower_bound = total_diffs  # differences are 1..M, so max entry >= total_diffs

    # --- Fast greedy baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct()
    best_scope = max(max(row) for row in best_solution)

    # --- Validation ---
    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k_plus_1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    # --- Simulated Annealing with Row Regeneration ---
    def sa_row_regeneration(rows, deadline):
        T = 10.0
        cooling = 0.995
        T_min = 0.1
        current_max = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        best_max = current_max

        used = get_used_diffs(rows)

        def generate_row(used_diffs, k_val, randomize=False, p_smallest=0.8, window=3, max_search=300):
            row = [0]
            current_row_diffs = set()
            for _ in range(k_val):
                candidates = []
                x = row[-1] + 1
                forbidden = used_diffs | current_row_diffs
                while x < max_search:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        candidates.append(x)
                        if len(candidates) >= window:
                            break
                    x += 1
                if not candidates:
                    return None
                if randomize:
                    if random.random() < p_smallest:
                        next_x = candidates[0]
                    else:
                        next_x = random.choice(candidates)
                else:
                    next_x = candidates[0]
                row.append(next_x)
                current_row_diffs |= {next_x - a for a in row[:-1]}
            return row

        while time.time() < deadline - 40:
            if T < T_min:
                T = 5.0
                cooling = 0.998

            r_idx = random.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            used -= old_diffs

            new_row = generate_row(used, k, randomize=True, p_smallest=0.8, window=3, max_search=best_max + 30)
            if new_row is None:
                used |= old_diffs
                continue

            new_max_val = max(new_row)
            other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
            new_global_max = max(other_max, new_max_val)
            delta = new_global_max - current_max

            if delta < 0 or random.random() < math.exp(-delta / T):
                rows[r_idx] = new_row
                used |= set(new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i + 1, len(new_row)))
                current_max = new_global_max
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
            else:
                used |= old_diffs
            T *= cooling

        return best_rows, best_max

    if time.time() < deadline - 45:
        best_solution, best_scope = sa_row_regeneration(best_solution, deadline)

    # --- Local Search with Element Moves (soft surrogate objective) ---
    def local_search(rows, deadline):
        current_rows = [list(r) for r in rows]
        diff_counts = {}
        current_conflict_count = 0

        def add_diff(d):
            nonlocal current_conflict_count
            if d in diff_counts:
                if diff_counts[d] == 1:
                    current_conflict_count += 1
                diff_counts[d] += 1
            else:
                diff_counts[d] = 1

        def remove_diff(d):
            nonlocal current_conflict_count
            if d in diff_counts:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                elif diff_counts[d] == 1:
                    current_conflict_count -= 1

        for r in current_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    add_diff(r[j] - r[i])

        current_scope = max(max(r) for r in current_rows)
        current_cost = current_scope + 1000 * current_conflict_count
        best_cost = current_cost
        best_rows_final = [list(r) for r in current_rows]
        best_scope_final = current_scope

        T = 5.0
        T_min = 0.1
        cooling = 0.999
        steps = 0

        while time.time() < deadline - 15:
            if T < T_min:
                T = 2.0
                cooling = 0.9995
                if steps > 80000:
                    break

            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            p_idx = random.randint(1, len(row) - 1)
            old_val = row[p_idx]

            diffs_to_remove = []
            for i in range(len(row)):
                if i != p_idx:
                    diffs_to_remove.append(abs(row[i] - old_val))

            for d in diffs_to_remove:
                remove_diff(d)
            row.pop(p_idx)

            new_val = None
            new_r_idx = r_idx

            if random.random() < 0.85:
                max_search = min(best_scope_final + 15, 300)
                found = False
                for _ in range(15):
                    cand = random.randint(1, max_search)
                    if cand in row:
                        continue
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = abs(cand - a)
                        if d in diff_counts:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        pos = 0
                        while pos < len(row) and row[pos] < cand:
                            pos += 1
                        row.insert(pos, cand)
                        for d in new_diffs:
                            add_diff(d)
                        new_val = cand
                        found = True
                        break
                if not found:
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for d in diffs_to_remove:
                        add_diff(d)
                    continue
            else:
                new_r_idx = random.randint(0, n_rows - 1)
                while new_r_idx == r_idx:
                    new_r_idx = random.randint(0, n_rows - 1)
                cand = old_val
                target_row = current_rows[new_r_idx]
                if cand in target_row:
                    cand = random.randint(1, best_scope_final + 15)
                    if cand in target_row:
                        pos = 0
                        while pos < len(row) and row[pos] < old_val:
                            pos += 1
                        row.insert(pos, old_val)
                        for d in diffs_to_remove:
                            add_diff(d)
                        continue
                valid = True
                new_diffs = []
                for a in target_row:
                    d = abs(cand - a)
                    if d in diff_counts:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    pos = 0
                    while pos < len(target_row) and target_row[pos] < cand:
                        pos += 1
                    target_row.insert(pos, cand)
                    for d in new_diffs:
                        add_diff(d)
                    new_val = cand
                else:
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for d in diffs_to_remove:
                        add_diff(d)
                    continue

            new_scope = max(max(r) for r in current_rows)
            new_cost = new_scope + 1000 * current_conflict_count
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                current_scope = new_scope
                if is_valid(current_rows):
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_scope_final = current_scope
                        best_rows_final = [list(r) for r in current_rows]
            else:
                if new_r_idx != r_idx:
                    target_row = current_rows[new_r_idx]
                    idx_new = target_row.index(new_val)
                    target_row.pop(idx_new)
                    for a in target_row:
                        remove_diff(abs(a - new_val))
                else:
                    row = current_rows[r_idx]
                    idx_new = row.index(new_val)
                    row.pop(idx_new)
                    for a in row:
                        remove_diff(abs(a - new_val))
                row = current_rows[r_idx]
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)

            T *= cooling
            steps += 1

        return best_rows_final, best_scope_final

    if time.time() < deadline - 20:
        best_solution, best_scope = local_search(best_solution, deadline)

    # --- Backtracking Search for Target Scope ---
    def backtracking_search(target_scope, deadline, seed_offset=0):
        rng = random.Random(seed_offset * 12345 + 678)
        used_diffs = [False] * (target_scope + 1)
        used_count = 0
        rows = []

        def build_row(pos, current_mark, row_so_far, depth):
            nonlocal used_count
            if time.time() > deadline:
                return None
            if pos == k_plus_1:
                return list(row_so_far)
            max_x = target_scope - (k_plus_1 - pos - 1)
            if max_x <= current_mark:
                return None
            valid_candidates = []
            for x in range(current_mark + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if used_diffs[d]:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
                if len(valid_candidates) >= 60:
                    break
            if not valid_candidates:
                return None
            n_smallest = min(20, len(valid_candidates))
            candidates_to_try = valid_candidates[:n_smallest]
            if len(valid_candidates) > n_smallest and rng.random() < 0.3:
                extra = rng.sample(valid_candidates[n_smallest:], min(10, len(valid_candidates) - n_smallest))
                candidates_to_try = candidates_to_try + extra
            rng.shuffle(candidates_to_try)
            for x in candidates_to_try:
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    new_diffs.append(d)
                    used_diffs[d] = True
                used_count += len(new_diffs)
                res = build_row(pos + 1, x, row_so_far + [x], depth + 1)
                if res is not None:
                    return res
                for d in new_diffs:
                    used_diffs[d] = False
                used_count -= len(new_diffs)
            return None

        for i in range(n_rows):
            row = build_row(1, 0, [0], 0)
            if row is None:
                # Undo
                for r in rows:
                    for j in range(len(r)):
                        for i2 in range(j + 1, len(r)):
                            used_diffs[r[i2] - r[j]] = False
                return None
            rows.append(row)
            for j in range(len(row)):
                for i2 in range(j + 1, len(row)):
                    d = row[i2] - row[j]
                    if not used_diffs[d]:
                        used_diffs[d] = True
                        used_count += 1
        return rows

    # Try backtracking for scopes below current best
    if time.time() < deadline - 10:
        target = best_scope - 1
        attempt = 0
        while target >= lower_bound and time.time() < deadline - 8:
            found = False
            for _ in range(30):
                if time.time() > deadline - 8:
                    break
                sol = backtracking_search(target, deadline, attempt)
                if sol and is_valid(sol):
                    new_scope = max(max(r) for r in sol)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = sol
                        found = True
                        # Polish it
                        if time.time() < deadline - 5:
                            polished, p_scope = local_search(best_solution, deadline)
                            if p_scope < best_scope:
                                best_scope = p_scope
                                best_solution = polished
                        break
                attempt += 1
            if found:
                target = best_scope - 1
            else:
                target -= 1
                if target < lower_bound:
                    break

    # --- Final Aggressive Compaction Polish ---
    def final_polish(rows, deadline):
        final_rows = [list(r) for r in rows]
        used_set = get_used_diffs(final_rows)

        for pass_num in range(30):
            if time.time() > deadline - 2:
                break
            improved = False
            for row_idx in range(n_rows):
                row = final_rows[row_idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline - 2:
                        break
                    val = row[pos]
                    if val <= pos:
                        continue
                    row.pop(pos)
                    removed_diffs = set()
                    for a in row:
                        d = abs(val - a)
                        removed_diffs.add(d)
                        used_set.discard(d)
                    new_val = None
                    for candidate in range(pos + 1, val):
                        if candidate in row:
                            continue
                        valid = True
                        for a in row:
                            d = abs(candidate - a)
                            if d == 0 or d in used_set:
                                valid = False
                                break
                        if valid:
                            new_val = candidate
                            break
                    if new_val is not None and new_val < val:
                        insert_pos = 0
                        while insert_pos < len(row) and row[insert_pos] < new_val:
                            insert_pos += 1
                        row.insert(insert_pos, new_val)
                        for a in row:
                            if a != new_val:
                                used_set.add(abs(new_val - a))
                        improved = True
                    else:
                        row.insert(pos, val)
                        for d in removed_diffs:
                            used_set.add(d)
            if not improved:
                break

        final_scope = max(max(r) for r in final_rows)
        if is_valid(final_rows) and final_scope < max(max(r) for r in rows):
            return final_rows, final_scope
        return rows, max(max(r) for r in rows)

    if time.time() < deadline - 1:
        best_solution, best_scope = final_polish(best_solution, deadline)

    # Final validity check
    if not is_valid(best_solution):
        # Fallback to greedy
        used = set()
        best_solution = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k_plus_1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_solution.append(row)

    return best_solution
