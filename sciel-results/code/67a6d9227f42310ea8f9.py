import time
import random

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. Aim to finish by 135s.
    deadline = start_time + 132.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i - 1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(rand_prob=0.1, window=20, seed=None):
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                limit = 250
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
            rows.append(row)
        return rows

    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs

    def local_search(rows, time_limit):
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        stuck_counter = 0

        while time.time() < start + time_limit:
            # 1. Targeted Descent: try to reduce the global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx

            if max_c_idx == 0:
                break

            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            all_diffs = get_all_diffs(best_rows)
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, len(row)):
                remove_set.add(row[j] - current_val)
            fixed_diffs = all_diffs - remove_set

            found_improvement = False
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                for i in range(max_c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(max_c_idx + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, len(row))]
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                row[max_c_idx] = new_val
                found_improvement = True
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    stuck_counter = 0
                break

            if found_improvement:
                continue

            # 2. Try other elements
            improved_any = False
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if r_idx == max_r_idx and c_idx == max_c_idx:
                        continue
                    curr_val = best_rows[r_idx][c_idx]
                    prev_v = best_rows[r_idx][c_idx - 1]
                    all_diffs = get_all_diffs(best_rows)
                    remove_set = set()
                    row = best_rows[r_idx]
                    for i in range(c_idx):
                        remove_set.add(curr_val - row[i])
                    for j in range(c_idx + 1, len(row)):
                        remove_set.add(row[j] - curr_val)
                    fixed_diffs = all_diffs - remove_set

                    for new_val in range(curr_val - 1, prev_v, -1):
                        valid = True
                        for i in range(c_idx):
                            d = new_val - row[i]
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        for j in range(c_idx + 1, len(row)):
                            d = row[j] - new_val
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        new_left = [new_val - row[i] for i in range(c_idx)]
                        new_right = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                        if len(set(new_left + new_right)) != len(new_left + new_right):
                            continue
                        row[c_idx] = new_val
                        improved_any = True
                        break
                    if improved_any:
                        break
                if improved_any:
                    break

            if improved_any:
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    stuck_counter = 0
                continue

            # 3. Perturbation: Row Reconstruction
            stuck_counter += 1
            r_idx = random.randint(0, n_rows - 1)
            other_rows = [best_rows[i] for i in range(n_rows) if i != r_idx]
            fixed_diffs = set()
            for r in other_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        fixed_diffs.add(r[j] - r[i])

            success = False
            for attempt in range(10):
                temp_row = [0]
                temp_used = set()
                while len(temp_row) < row_len:
                    last = temp_row[-1]
                    candidates = []
                    x = last + 1
                    limit = min(best_scope + 20, last + 80)
                    while x <= limit and len(candidates) < 30:
                        valid = True
                        for a in temp_row:
                            d = x - a
                            if d in fixed_diffs or d in temp_used:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                        x += 1
                    if not candidates:
                        while True:
                            x += 1
                            valid = True
                            for a in temp_row:
                                d = x - a
                                if d in fixed_diffs or d in temp_used:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(x)
                                break
                    if random.random() < 0.3 and len(candidates) > 1:
                        x = random.choice(candidates)
                    else:
                        x = candidates[0]
                    temp_row.append(x)
                    for a in temp_row[:-1]:
                        temp_used.add(x - a)
                if max(temp_row) <= best_scope:
                    best_rows[r_idx] = temp_row
                    success = True
                    break

            if success:
                stuck_counter = 0
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope

        return best_rows, best_scope

    # Phase 1: Diverse Initialization
    best_rows = None
    best_scope = float('inf')
    num_inits = 15
    init_time_limit = 6.0
    start_init = time.time()

    for i in range(num_inits):
        if time.time() > start_init + init_time_limit:
            break
        rows = greedy_init(rand_prob=0.05 + 0.01 * i, window=25, seed=i)
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    # Phase 2: Local Search Refinement
    local_time = 55.0
    if time.time() > deadline - 65:
        local_time = deadline - time.time() - 10

    if local_time > 10 and best_rows:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 2.5: High-precision polish pass
    if best_rows and time.time() < deadline - 20:
        polish_time = min(15.0, deadline - time.time() - 10)
        if polish_time > 3:
            polish_rows = [r[:] for r in best_rows]
            polish_scope = best_scope

            def polish_pass(rows, scope, time_limit):
                start_p = time.time()
                improved = True
                while improved and time.time() < start_p + time_limit:
                    improved = False
                    for r_idx in range(n_rows):
                        for c_idx in range(1, row_len):
                            if time.time() > start_p + time_limit:
                                return rows, scope
                            curr_val = rows[r_idx][c_idx]
                            prev_v = rows[r_idx][c_idx - 1]

                            all_diffs = set()
                            for r in rows:
                                for i in range(len(r)):
                                    for j in range(i + 1, len(r)):
                                        all_diffs.add(r[j] - r[i])

                            remove_set = set()
                            row = rows[r_idx]
                            for i in range(c_idx):
                                remove_set.add(curr_val - row[i])
                            for j in range(c_idx + 1, len(row)):
                                remove_set.add(row[j] - curr_val)
                            fixed_diffs = all_diffs - remove_set

                            for new_val in range(curr_val - 1, prev_v, -1):
                                valid = True
                                for i in range(c_idx):
                                    d = new_val - row[i]
                                    if d in fixed_diffs:
                                        valid = False
                                        break
                                if not valid:
                                    continue
                                for j in range(c_idx + 1, len(row)):
                                    d = row[j] - new_val
                                    if d in fixed_diffs:
                                        valid = False
                                        break
                                if not valid:
                                    continue
                                new_left = [new_val - row[i] for i in range(c_idx)]
                                new_right = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                                if len(set(new_left + new_right)) != len(new_left + new_right):
                                    continue
                                row[c_idx] = new_val
                                improved = True
                                new_scope = max(max(r) for r in rows)
                                if new_scope < scope:
                                    scope = new_scope
                                break
                return rows, scope

            polish_rows, polish_scope = polish_pass(polish_rows, polish_scope, polish_time)
            if validate(polish_rows):
                if polish_scope < best_scope:
                    best_rows = polish_rows
                    best_scope = polish_scope

    # Phase 3: Backtracking to try to beat the local search result
    current_target = best_scope - 1
    seed_counter = 0

    def backtracking_search(n_rows, k, target_scope, time_limit, seed=0):
        random.seed(seed)
        start = time.time()
        rows = [[0] * row_len for _ in range(n_rows)]
        used_diffs = set()

        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            if row_idx == n_rows:
                return True
            if pos == row_len:
                return solve(row_idx + 1, 1)

            prev_val = rows[row_idx][pos - 1]
            lower = prev_val + 1
            upper = target_scope
            if lower > upper:
                return False

            candidates = []
            scan_limit = min(upper, lower + 80)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 15:
                        break

            if not candidates:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 15:
                            break

            if not candidates:
                return False

            random.shuffle(candidates)
            candidates.sort()

            for val in candidates:
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                rows[row_idx][pos] = val
                if solve(row_idx, pos + 1):
                    return True
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    while time.time() < deadline - 5:
        if current_target < 112:
            break
        search_time = 4.0
        if time.time() > deadline - 12:
            search_time = deadline - time.time() - 2
        if search_time < 1.0:
            break

        result = backtracking_search(n_rows, k, current_target, search_time, seed=seed_counter)
        seed_counter += 1
        if result:
            if validate(result):
                new_scope = get_scope(result)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = result
                    current_target = best_scope - 1
                else:
                    break
            else:
                break
        else:
            if time.time() < deadline - 8:
                result_retry = backtracking_search(n_rows, k, current_target, 4.0, seed=seed_counter)
                seed_counter += 1
                if result_retry and validate(result_retry):
                    new_scope = get_scope(result_retry)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = result_retry
                        current_target = best_scope - 1
                    else:
                        break
                else:
                    break
            else:
                break

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
