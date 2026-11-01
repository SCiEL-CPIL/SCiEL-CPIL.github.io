import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0

    row_len = k + 1
    total_diffs = n_rows * k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def all_diffs_set(rows):
        s = set()
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    s.add(r[j] - r[i])
        return s

    def greedy_init(rand_prob=0.05, window=15):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                limit = 400
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
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

    def backtracking_row_init(used_diffs, max_search=500):
        """Try to build a row using backtracking for tighter packing."""
        row = [0]
        row_diffs = set()
        
        def backtrack(pos):
            if pos == row_len:
                return True
            last = row[-1]
            for x in range(last + 1, max_search + 1):
                new_diffs = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in used_diffs or d in row_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                row.append(x)
                row_diffs.update(new_diffs)
                if backtrack(pos + 1):
                    return True
                row.pop()
                row_diffs -= new_diffs
            return False
        
        if backtrack(1):
            return row
        return None

    def backtracking_init(max_search=300):
        """Initialize rows using backtracking for each row."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            # Try backtracking first
            row = backtracking_row_init(used_diffs, max_search)
            if row is None:
                # Fall back to greedy for this row
                row = [0]
                while len(row) < row_len:
                    last = row[-1]
                    found = False
                    for x in range(last + 1, max_search + 200):
                        ok = True
                        for a in row:
                            if (x - a) in used_diffs:
                                ok = False
                                break
                        if ok:
                            row.append(x)
                            for a in row[:-1]:
                                used_diffs.add(x - a)
                            found = True
                            break
                    if not found:
                        # Extend search
                        for x in range(max_search + 200, max_search + 500):
                            ok = True
                            for a in row:
                                if (x - a) in used_diffs:
                                    ok = False
                                    break
                            if ok:
                                row.append(x)
                                for a in row[:-1]:
                                    used_diffs.add(x - a)
                                found = True
                                break
                        if not found:
                            return None
            else:
                for i in range(row_len):
                    for j in range(i):
                        used_diffs.add(row[i] - row[j])
            rows.append(row)
        return rows

    def global_max_targeted_descent(rows, time_limit):
        """
        Always target the single largest element in the entire configuration.
        This directly attacks the scope objective.
        """
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        max_passes = 50
        pass_count = 0
        
        while improved and pass_count < max_passes and (time.time() - start) < time_limit:
            improved = False
            pass_count += 1
            
            # Find the global maximum element
            max_val = -1
            max_r = -1
            max_c = -1
            for r_idx, r in enumerate(rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r = r_idx
                        max_c = c_idx
            
            if max_c == 0:
                break  # Can't reduce 0
            
            row = rows[max_r]
            curr_val = max_val
            prev_val = row[max_c - 1]
            
            # Build fixed diffs (all diffs not involving this element)
            all_diffs = all_diffs_set(rows)
            remove_set = set()
            for i in range(max_c):
                remove_set.add(curr_val - row[i])
            for j in range(max_c + 1, row_len):
                remove_set.add(row[j] - curr_val)
            fixed_diffs = all_diffs - remove_set
            
            # Try to reduce
            found = False
            for new_val in range(curr_val - 1, prev_val, -1):
                ok = True
                new_diffs = []
                for i in range(max_c):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                for j in range(max_c + 1, row_len):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                row[max_c] = new_val
                found = True
                improved = True
                break
            
            if not found:
                # Try other elements too, but prioritize larger ones
                improved_any = False
                for r_idx in range(n_rows):
                    for c_idx in range(row_len - 1, 0, -1):
                        if r_idx == max_r and c_idx == max_c:
                            continue
                        curr = rows[r_idx][c_idx]
                        prev = rows[r_idx][c_idx - 1]
                        all_diffs = all_diffs_set(rows)
                        remove_set = set()
                        r = rows[r_idx]
                        for i in range(c_idx):
                            remove_set.add(curr - r[i])
                        for j in range(c_idx + 1, row_len):
                            remove_set.add(r[j] - curr)
                        fixed_diffs = all_diffs - remove_set
                        
                        found = False
                        for new_val in range(curr - 1, prev, -1):
                            ok = True
                            new_diffs = []
                            for i in range(c_idx):
                                d = new_val - r[i]
                                if d in fixed_diffs:
                                    ok = False
                                    break
                                new_diffs.append(d)
                            if not ok:
                                continue
                            for j in range(c_idx + 1, row_len):
                                d = r[j] - new_val
                                if d in fixed_diffs:
                                    ok = False
                                    break
                                new_diffs.append(d)
                            if not ok:
                                continue
                            if len(set(new_diffs)) != len(new_diffs):
                                continue
                            r[c_idx] = new_val
                            found = True
                            improved_any = True
                            improved = True
                            break
                        if found:
                            break
                    if improved_any:
                        break
                if not improved_any:
                    break

        return rows

    def targeted_descent_all(rows, time_limit):
        """Full targeted descent on all elements, right to left."""
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        pass_count = 0
        max_passes = 30
        while improved and pass_count < max_passes and (time.time() - start) < time_limit:
            improved = False
            pass_count += 1
            for r_idx in range(n_rows):
                if (time.time() - start) > time_limit:
                    break
                row = rows[r_idx]
                other_diffs = set()
                for j in range(n_rows):
                    if j != r_idx:
                        for i in range(row_len):
                            for jj in range(i + 1, row_len):
                                other_diffs.add(rows[j][jj] - rows[j][i])
                for c_idx in range(row_len - 1, 0, -1):
                    if (time.time() - start) > time_limit:
                        break
                    curr_val = row[c_idx]
                    prev_val = row[c_idx - 1]
                    fixed = set(other_diffs)
                    for i in range(row_len):
                        if i == c_idx:
                            continue
                        for j in range(i + 1, row_len):
                            if j == c_idx:
                                continue
                            fixed.add(row[j] - row[i])
                    found = False
                    for new_val in range(curr_val - 1, prev_val, -1):
                        ok = True
                        new_diffs = []
                        for i in range(c_idx):
                            d = new_val - row[i]
                            if d in fixed:
                                ok = False
                                break
                            new_diffs.append(d)
                        if not ok:
                            continue
                        for j in range(c_idx + 1, row_len):
                            d = row[j] - new_val
                            if d in fixed:
                                ok = False
                                break
                            new_diffs.append(d)
                        if not ok:
                            continue
                        if len(set(new_diffs)) != len(new_diffs):
                            continue
                        row[c_idx] = new_val
                        found = True
                        improved = True
                        break
        return rows

    def simulated_annealing(rows, time_limit):
        """SA with element moves, conflict-aware."""
        rows = [list(r) for r in rows]
        start = time.time()
        current_conflicts = count_conflicts(rows)
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_conflicts = current_conflicts
        best_scope = current_scope
        T = 20.0
        T_min = 0.01
        cooling = 0.998
        no_improve = 0
        max_no_improve = 5000

        while (time.time() - start) < time_limit and T > T_min and no_improve < max_no_improve:
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, row_len - 1)
            row = rows[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            upper = row[pos + 1] - 1 if pos < row_len - 1 else current_scope + 20
            if lower > upper:
                continue
            v = random.randint(lower, upper)
            if v == old_val:
                continue
            row[pos] = v
            new_conflicts = count_conflicts(rows)
            new_scope = get_scope(rows)
            delta_c = new_conflicts - current_conflicts
            delta_s = new_scope - current_scope
            accept = False
            if new_conflicts < current_conflicts:
                accept = True
            elif new_conflicts == current_conflicts:
                if new_scope < current_scope:
                    accept = True
                elif random.random() < math.exp(-delta_s / max(T, 0.01)):
                    accept = True
            else:
                if random.random() < math.exp(-(delta_c * 2 + delta_s) / max(T, 0.01)):
                    accept = True
            if accept:
                current_conflicts = new_conflicts
                current_scope = new_scope
                if (new_conflicts < best_conflicts or
                    (new_conflicts == best_conflicts and new_scope < best_scope)):
                    best_conflicts = new_conflicts
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                row[pos] = old_val
                no_improve += 1
            T *= cooling
        return best_rows

    def rebuild_row(used_other, max_val=None, jitter=0.0):
        row = [0]
        row_diffs = set()
        while len(row) < row_len:
            last = row[-1]
            found = False
            search_end = max_val if max_val else 500
            for trial in range(last + 1, search_end + 1):
                new_diffs = {trial - a for a in row}
                if not (new_diffs & used_other) and not (new_diffs & row_diffs):
                    if jitter > 0 and random.random() < jitter:
                        continue
                    row.append(trial)
                    row_diffs |= new_diffs
                    found = True
                    break
            if not found:
                for trial in range(search_end + 1, search_end + 200):
                    new_diffs = {trial - a for a in row}
                    if not (new_diffs & used_other) and not (new_diffs & row_diffs):
                        row.append(trial)
                        row_diffs |= new_diffs
                        found = True
                        break
            if not found:
                return None
        return row

    def row_polish(rows, time_limit):
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        pass_count = 0
        max_passes = 5
        while improved and pass_count < max_passes and (time.time() - start) < time_limit:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if (time.time() - start) > time_limit:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        for a in range(row_len):
                            for b in range(a + 1, row_len):
                                used_other.add(rows[j][b] - rows[j][a])
                best_max = max(rows[i])
                new_row = rebuild_row(used_other, max_val=best_max - 1, jitter=0.0)
                if new_row is not None and max(new_row) < best_max:
                    rows[i] = new_row
                    best_max = max(new_row)
                    improved = True
                for att in range(5):
                    if (time.time() - start) > time_limit:
                        break
                    jitter = 0.15 + 0.1 * att
                    new_row = rebuild_row(used_other, max_val=best_max - 1, jitter=jitter)
                    if new_row is not None and max(new_row) < best_max:
                        rows[i] = new_row
                        best_max = max(new_row)
                        improved = True
        return rows

    def high_precision_polish(rows, time_limit):
        """
        Exhaustive single-element optimization: try every possible value
        for each element (within bounds) to find any remaining improvement.
        Used only in final phase.
        """
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        max_passes = 10
        pass_count = 0
        while improved and pass_count < max_passes and (time.time() - start) < time_limit:
            improved = False
            pass_count += 1
            for r_idx in range(n_rows):
                if (time.time() - start) > time_limit:
                    break
                for c_idx in range(row_len - 1, 0, -1):
                    if (time.time() - start) > time_limit:
                        break
                    row = rows[r_idx]
                    curr_val = row[c_idx]
                    prev_val = row[c_idx - 1]
                    next_val = row[c_idx + 1] if c_idx < row_len - 1 else curr_val + 100
                    
                    all_diffs = all_diffs_set(rows)
                    remove_set = set()
                    for i in range(c_idx):
                        remove_set.add(curr_val - row[i])
                    for j in range(c_idx + 1, row_len):
                        remove_set.add(row[j] - curr_val)
                    fixed_diffs = all_diffs - remove_set
                    
                    # Try every value from curr_val-1 down to prev_val+1
                    for new_val in range(curr_val - 1, prev_val, -1):
                        if new_val >= next_val:
                            continue
                        ok = True
                        new_diffs = []
                        for i in range(c_idx):
                            d = new_val - row[i]
                            if d in fixed_diffs:
                                ok = False
                                break
                            new_diffs.append(d)
                        if not ok:
                            continue
                        for j in range(c_idx + 1, row_len):
                            d = row[j] - new_val
                            if d in fixed_diffs:
                                ok = False
                                break
                            new_diffs.append(d)
                        if not ok:
                            continue
                        if len(set(new_diffs)) != len(new_diffs):
                            continue
                        row[c_idx] = new_val
                        improved = True
                        break
        return rows

    def kick_and_polish(rows, time_limit):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    for a in range(row_len):
                        for b in range(a + 1, row_len):
                            used_other.add(rows[j][b] - rows[j][a])
            success = False
            for att in range(10):
                jitter = 0.2 + 0.1 * att
                new_row = rebuild_row(used_other, max_val=500, jitter=jitter)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, max_val=500, jitter=0.0)
                if new_row is not None:
                    rows[idx] = new_row
        rows = global_max_targeted_descent(rows, min(3.0, time_limit * 0.4))
        rows = row_polish(rows, min(3.0, time_limit * 0.4))
        return rows

    # --- MAIN SEARCH ---
    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')

    # Phase 1: Multiple restarts with diverse initialization
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        # Alternate between greedy and backtracking init
        if i % 2 == 0:
            rand_prob = 0.03 + 0.04 * (i // 2)
            window = 10 + 4 * (i // 2)
            rows = greedy_init(rand_prob, window)
        else:
            max_search = 200 + 50 * (i // 2)
            rows = backtracking_init(max_search)
            if rows is None:
                rows = greedy_init(0.1, 20)
        
        scope = get_scope(rows)
        conflicts = count_conflicts(rows)

        # Global max targeted descent (primary refinement)
        t1_rows = global_max_targeted_descent(rows, 4.0)
        t1_conflicts = count_conflicts(t1_rows)
        t1_scope = get_scope(t1_rows)
        if t1_conflicts < best_conflicts or (t1_conflicts == best_conflicts and t1_scope < best_scope):
            best_conflicts = t1_conflicts
            best_scope = t1_scope
            best_rows = t1_rows

        if time.time() > deadline - 25:
            break

        # Full targeted descent
        t1b_rows = targeted_descent_all(t1_rows, 3.0)
        t1b_conflicts = count_conflicts(t1b_rows)
        t1b_scope = get_scope(t1b_rows)
        if t1b_conflicts < best_conflicts or (t1b_conflicts == best_conflicts and t1b_scope < best_scope):
            best_conflicts = t1b_conflicts
            best_scope = t1b_scope
            best_rows = t1b_rows

        if time.time() > deadline - 25:
            break

        # SA
        t2_rows = simulated_annealing(t1b_rows, 4.0)
        t2_conflicts = count_conflicts(t2_rows)
        t2_scope = get_scope(t2_rows)
        if t2_conflicts < best_conflicts or (t2_conflicts == best_conflicts and t2_scope < best_scope):
            best_conflicts = t2_conflicts
            best_scope = t2_scope
            best_rows = t2_rows

        if best_conflicts == 0 and best_scope < 118:
            break

    if best_rows is None:
        best_rows = greedy_init(0.1, 15)
        best_rows = global_max_targeted_descent(best_rows, 10.0)
        best_scope = get_scope(best_rows)
        best_conflicts = count_conflicts(best_rows)

    # Phase 2: ILS with kick + polish
    remaining = deadline - time.time() - 10
    if remaining > 15:
        num_iters = int(remaining / 8.0)
        num_iters = max(1, num_iters)
        for it in range(num_iters):
            if time.time() > deadline - 10:
                break
            it_time = min(8.0, deadline - time.time() - 10)
            if it_time < 3:
                break
            cur = kick_and_polish(best_rows, it_time)
            cur_scope = get_scope(cur)
            cur_conflicts = count_conflicts(cur)
            if cur_conflicts < best_conflicts or (cur_conflicts == best_conflicts and cur_scope < best_scope):
                best_conflicts = cur_conflicts
                best_scope = cur_scope
                best_rows = cur

    # Phase 3: Final polish with high-precision exhaustive sweep
    remaining = deadline - time.time() - 3
    if remaining > 5:
        f1 = global_max_targeted_descent(best_rows, min(remaining * 0.4, 10.0))
        f1_conflicts = count_conflicts(f1)
        f1_scope = get_scope(f1)
        if f1_conflicts < best_conflicts or (f1_conflicts == best_conflicts and f1_scope < best_scope):
            best_conflicts = f1_conflicts
            best_scope = f1_scope
            best_rows = f1

        remaining2 = deadline - time.time() - 2
        if remaining2 > 3:
            f2 = row_polish(best_rows, min(remaining2 * 0.4, 8.0))
            f2_conflicts = count_conflicts(f2)
            f2_scope = get_scope(f2)
            if f2_conflicts < best_conflicts or (f2_conflicts == best_conflicts and f2_scope < best_scope):
                best_conflicts = f2_conflicts
                best_scope = f2_scope
                best_rows = f2

        # High-precision exhaustive polish (final)
        remaining3 = deadline - time.time() - 1
        if remaining3 > 2:
            f3 = high_precision_polish(best_rows, remaining3)
            f3_conflicts = count_conflicts(f3)
            f3_scope = get_scope(f3)
            if f3_conflicts < best_conflicts or (f3_conflicts == best_conflicts and f3_scope < best_scope):
                best_conflicts = f3_conflicts
                best_scope = f3_scope
                best_rows = f3

    # Final global max descent sweep
    remaining = deadline - time.time() - 1
    if remaining > 1:
        best_rows = global_max_targeted_descent(best_rows, remaining)

    return best_rows
