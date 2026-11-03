import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    num_entries = k + 1
    # number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    # counting lower bound on scope M (differences in [1, M-1])
    lb = total_diffs + 1

    # We track used differences in a bytearray of size (lb + 4000) to allow slack.
    # A difference d is "used" iff used_arr[d] == 1.
    arr_size = lb + 4000

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != num_entries:
                return False
            if r[0] != 0:
                return False
            for i in range(num_entries):
                ai = r[i]
                for j in range(i + 1, num_entries):
                    d = r[j] - ai
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_set(rows):
        used = set()
        for r in rows:
            for i in range(num_entries):
                ai = r[i]
                for j in range(i + 1, num_entries):
                    used.add(r[j] - ai)
        return used

    def greedy_construct(seed_rng):
        """Deterministic-ish greedy: build rows one by one, picking smallest valid x."""
        used = bytearray(arr_size)
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        d = x - a
                        if d >= arr_size or used[d]:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used[x - a] = 1
                        row.append(x)
                        break
                    x += 1
            rows.append(row)
        return rows

    def generate_row(used, k_val, randomize=False, p_smallest=0.7, window=5, max_search=300):
        """Generate a single row given the used bytearray."""
        row = [0]
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            while x < max_search:
                ok = True
                for a in row:
                    d = x - a
                    if d >= arr_size or used[d]:
                        ok = False
                        break
                if ok:
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
            for a in row:
                used[next_x - a] = 1
            row.append(next_x)
        return row

    def remove_row(used, row):
        for i in range(num_entries):
            ai = row[i]
            for j in range(i + 1, num_entries):
                used[row[j] - ai] = 0

    def add_row(used, row):
        for i in range(num_entries):
            ai = row[i]
            for j in range(i + 1, num_entries):
                used[row[j] - ai] = 1

    def scope_of(rows):
        m = 0
        for r in rows:
            if r[-1] > m:
                m = r[-1]
        return m

    # ---- 1. Initial greedy solution ----
    rows = greedy_construct(None)
    used = bytearray(arr_size)
    for r in rows:
        add_row(used, r)

    best_rows = [r[:] for r in rows]
    best_max = scope_of(best_rows)

    # ---- 2. Simulated Annealing with row regeneration + 2-element swaps ----
    T = 8.0
    cooling = 0.995
    T_min = 0.1
    current_max = best_max

    def sa_row_regenerate():
        nonlocal T, current_max, best_max, best_rows
        while time.time() < deadline - 60:
            if T < T_min:
                T = 5.0
                cooling = 0.998

            r_idx = random.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            remove_row(used, old_row)

            new_row = generate_row(used, k, randomize=True, p_smallest=0.8,
                                   window=3, max_search=best_max + 25)
            if new_row is None:
                add_row(used, old_row)
                continue

            new_max_val = max(new_row)
            other_max = 0
            for i in range(n_rows):
                if i != r_idx and rows[i][-1] > other_max:
                    other_max = rows[i][-1]
            new_global_max = other_max if other_max > new_max_val else new_max_val
            delta = new_global_max - current_max

            if delta < 0 or random.random() < math.exp(-delta / T):
                rows[r_idx] = new_row
                add_row(used, new_row)
                current_max = new_global_max
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
            else:
                add_row(used, old_row)
            T *= cooling

    def sa_two_element_swap():
        nonlocal T, current_max, best_max, best_rows
        while time.time() < deadline - 60:
            if T < T_min:
                T = 4.0
                cooling = 0.999

            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            p = random.randint(1, num_entries - 1)
            old_val = row[p]

            # remove old_val's diffs
            for i in range(num_entries):
                if i != p:
                    used[abs(row[i] - old_val)] = 0

            row.pop(p)

            # try random new value in same row
            max_search = min(best_max + 15, arr_size - 1)
            found = False
            for _ in range(12):
                cand = random.randint(1, max_search)
                if cand in row:
                    continue
                ok = True
                for a in row:
                    d = abs(cand - a)
                    if d >= arr_size or used[d]:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used[abs(cand - a)] = 1
                    pos = 0
                    while pos < len(row) and row[pos] < cand:
                        pos += 1
                    row.insert(pos, cand)
                    found = True
                    break
            if not found:
                # revert
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for i in range(num_entries):
                    if i != p:
                        used[abs(row[i] - old_val)] = 1
                continue

            new_scope = scope_of(rows)
            delta = new_scope - current_max
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_max = new_scope
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
            else:
                # revert
                q = row.index(cand)
                row.pop(q)
                for a in row:
                    used[abs(a - cand)] = 0
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for i in range(num_entries):
                    if i != p:
                        used[abs(row[i] - old_val)] = 1
            T *= cooling

    # Alternate phases to use the budget
    phase_budget = 20
    for _ in range(6):
        if time.time() > deadline - 60:
            break
        sa_row_regenerate()
        if time.time() > deadline - 60:
            break
        sa_two_element_swap()

    # ---- 3. Scope-reduction polish (greedy re-insert at smallest valid) ----
    def polish_scope(rows_in):
        rows_p = [list(r) for r in rows_in]
        used_p = bytearray(arr_size)
        for r in rows_p:
            add_row(used_p, r)

        improved = True
        passes = 0
        while improved and passes < 40 and time.time() < deadline - 12:
            improved = False
            passes += 1
            for r_idx in range(n_rows):
                row = rows_p[r_idx]
                # process from the end (largest) to smallest
                for p in range(num_entries - 1, 0, -1):
                    old_val = row[p]
                    # remove
                    for i in range(num_entries):
                        if i != p:
                            used_p[abs(row[i] - old_val)] = 0
                    row.pop(p)
                    # find smallest valid position
                    best_x = None
                    for x in range(1, old_val):
                        if x in row:
                            continue
                        ok = True
                        for a in row:
                            d = abs(x - a)
                            if d >= arr_size or used_p[d]:
                                ok = False
                                break
                        if ok:
                            best_x = x
                            break
                    if best_x is None:
                        best_x = old_val
                    # insert
                    pos = 0
                    while pos < len(row) and row[pos] < best_x:
                        pos += 1
                    row.insert(pos, best_x)
                    for a in row:
                        if a != best_x:
                            used_p[abs(a - best_x)] = 1
                    if best_x < old_val:
                        improved = True
        return rows_p

    best_rows = polish_scope(best_rows)
    best_max = scope_of(best_rows)

    # ---- 4. Random restarts: re-greedy with randomization + quick SA + polish ----
    restart_budget = deadline - 40
    attempts = 0
    while time.time() < restart_budget and attempts < 6:
        attempts += 1
        r_rows = []
        r_used = bytearray(arr_size)
        ok_all = True
        for _ in range(n_rows):
            rr = generate_row(r_used, k, randomize=True, p_smallest=0.6,
                              window=6, max_search=best_max + 30)
            if rr is None:
                ok_all = False
                break
            r_rows.append(rr)
        if not ok_all:
            continue
        for rr in r_rows:
            add_row(r_used, rr)
        r_scope = scope_of(r_rows)
        if r_scope >= best_max:
            continue
        # quick polish
        r_rows = polish_scope(r_rows)
        r_scope = scope_of(r_rows)
        if r_scope < best_max and is_valid(r_rows):
            best_max = r_scope
            best_rows = [r[:] for r in r_rows]

    # ---- 5. Final validation + fallback ----
    if not is_valid(best_rows):
        # emergency deterministic fallback
        used_f = bytearray(arr_size)
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        d = x - a
                        if d >= arr_size or used_f[d]:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used_f[x - a] = 1
                        row.append(x)
                        break
                    x += 1
            best_rows.append(row)

    return best_rows
