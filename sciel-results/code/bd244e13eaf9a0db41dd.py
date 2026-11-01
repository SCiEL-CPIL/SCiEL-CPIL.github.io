import time
import random
import math
import heapq

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

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
                    if d <= 0 or d in used:
                        return False
                    used.add(d)
        return True

    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs

    def get_row_diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    # ---- Greedy initialization with multi-candidate trial ----
    def greedy_init(rand_prob=0.1, window=30, seed=None):
        rng = random.Random(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                limit = 300
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
                    x = last + 1
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
                if rng.random() < rand_prob and len(candidates) > 1:
                    x = rng.choice(candidates)
                else:
                    x = candidates[0]
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
            rows.append(row)
        return rows

    # ---- Exact local polish: systematically try all lower values for each position ----
    def exact_local_polish(rows, time_limit, seed=None):
        rng = random.Random(seed)
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        def try_decrease_row(row, row_idx, fixed_diffs, scope_limit):
            """Try to decrease any element in the row to reduce scope."""
            changed = True
            while changed and time.time() < start + time_limit:
                changed = False
                # Find the max element in this row
                max_val = -1
                max_c = -1
                for c_idx, val in enumerate(row):
                    if val > max_val:
                        max_val = val
                        max_c = c_idx
                if max_c == 0:
                    break
                prev_val = row[max_c - 1]
                # Compute diffs that involve max_val
                remove_set = set()
                for i in range(max_c):
                    remove_set.add(max_val - row[i])
                for j in range(max_c + 1, len(row)):
                    remove_set.add(row[j] - max_val)
                # Try all lower values
                for new_val in range(max_val - 1, prev_val, -1):
                    ok = True
                    # Check against fixed diffs
                    for i in range(max_c):
                        if (new_val - row[i]) in fixed_diffs:
                            ok = False
                            break
                    if not ok:
                        continue
                    for j in range(max_c + 1, len(row)):
                        if (row[j] - new_val) in fixed_diffs:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Check internal consistency
                    nd = [new_val - row[i] for i in range(max_c)] + [row[j] - new_val for j in range(max_c + 1, len(row))]
                    if len(set(nd)) != len(nd):
                        continue
                    # Also check that new_val - row[i] != row[j] - new_val for any i,j
                    for i in range(max_c):
                        for j in range(max_c + 1, len(row)):
                            if (new_val - row[i]) == (row[j] - new_val):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    row[max_c] = new_val
                    changed = True
                    break
            return row

        def full_polish_pass(rows, scope_limit):
            """Do a full pass trying to decrease all rows."""
            all_diffs = get_all_diffs(rows)
            for r_idx in range(n_rows):
                if time.time() > start + time_limit:
                    break
                row = rows[r_idx]
                row_diffs = get_row_diffs(row)
                fixed_diffs = all_diffs - row_diffs
                # Update all_diffs as we go
                old_diffs = row_diffs
                new_row = try_decrease_row(row, r_idx, fixed_diffs, scope_limit)
                new_diffs = get_row_diffs(new_row)
                # Update global diffs
                for d in old_diffs:
                    if d not in new_diffs:
                        all_diffs.discard(d)
                for d in new_diffs:
                    if d not in old_diffs:
                        all_diffs.add(d)
            return rows

        # Iterative polish passes
        no_improve = 0
        while time.time() < start + time_limit and no_improve < 5:
            old_scope = get_scope(best_rows)
            best_rows = full_polish_pass(best_rows, old_scope)
            new_scope = get_scope(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                no_improve = 0
            else:
                no_improve += 1
                # Perturb: reconstruct a random row
                if time.time() > start + time_limit:
                    break
                r_idx = rng.randint(0, n_rows - 1)
                all_diffs = get_all_diffs(best_rows)
                row = best_rows[r_idx]
                remove_set = get_row_diffs(row)
                fixed_diffs = all_diffs - remove_set

                new_row = [0]
                success = True
                while len(new_row) < row_len:
                    last = new_row[-1]
                    cands = []
                    x = last + 1
                    limit = best_scope + 20
                    while x <= limit and len(cands) < 40:
                        ok = True
                        for a in new_row:
                            if (x - a) in fixed_diffs:
                                ok = False
                                break
                        if ok:
                            cands.append(x)
                        x += 1
                    if not cands:
                        success = False
                        break
                    if rng.random() < 0.3 and len(cands) > 1:
                        x = rng.choice(cands)
                    else:
                        x = cands[0]
                    new_row.append(x)
                    for a in new_row[:-1]:
                        fixed_diffs.add(x - a)
                if success:
                    best_rows[r_idx] = new_row
                    # Quick polish after perturbation
                    best_rows = full_polish_pass(best_rows, best_scope + 10)
                    ns = get_scope(best_rows)
                    if ns < best_scope:
                        best_scope = ns
                        no_improve = 0
                    else:
                        no_improve += 1

        return best_rows, best_scope

    # ---- Beam search over rows with forward checking ----
    def beam_search_rows(target_scope, beam_width, time_limit, seed=0):
        rng = random.Random(seed)
        start = time.time()

        def generate_rows_for_used(used_diffs, target_scope, max_rows=100, rng=None):
            """Generate up to max_rows valid rows compatible with used_diffs."""
            rows = []
            row = [0]

            def dfs_row(pos):
                if time.time() > start + time_limit:
                    return
                if len(rows) >= max_rows:
                    return
                if pos == row_len:
                    rows.append(row[:])
                    return
                prev_val = row[-1]
                lower = prev_val + 1
                upper = target_scope
                for val in range(lower, upper + 1):
                    valid = True
                    for i in range(len(row)):
                        d = val - row[i]
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_diffs = [val - row[i] for i in range(len(row))]
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    row.append(val)
                    dfs_row(pos + 1)
                    row.pop()

            dfs_row(1)
            return rows

        beam = [([], set())]

        for row_idx in range(n_rows):
            new_beam = []
            for rows, used in beam:
                if time.time() > start + time_limit:
                    return None
                candidates = generate_rows_for_used(used, target_scope, max_rows=50, rng=rng)
                for cand in candidates:
                    new_used = used.copy()
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            new_used.add(cand[j] - cand[i])
                    new_rows = rows + [cand]
                    new_beam.append((new_rows, new_used))

            if not new_beam:
                return None

            new_beam.sort(key=lambda x: get_scope(x[0]))
            beam = new_beam[:beam_width]

        if beam:
            return beam[0][0]
        return None

    # ---- Phase 1: diverse greedy inits ----
    best_rows = None
    best_scope = float('inf')
    num_inits = 15
    init_budget = 8.0
    t0 = time.time()
    for i in range(num_inits):
        if time.time() > t0 + init_budget:
            break
        rows = greedy_init(rand_prob=0.05 + 0.01 * i, window=30, seed=i * 1000 + 7)
        sc = get_scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = [r[:] for r in rows]

    # ---- Phase 2: exact local polish ----
    if best_rows is not None:
        remaining = deadline - time.time()
        ls_budget = max(30.0, remaining * 0.4)
        if ls_budget > 10:
            per = max(8.0, ls_budget / 4)
            for s in range(4):
                if time.time() > deadline - 20:
                    break
                if s == 0:
                    src = best_rows
                else:
                    src = greedy_init(rand_prob=0.2 + 0.05 * s, window=40, seed=5000 + s * 31)
                rr, sc = exact_local_polish(src, per, seed=9000 + s * 17)
                if validate(rr) and sc < best_scope:
                    best_scope = sc
                    best_rows = rr

    # ---- Phase 3: beam search to try to beat current scope ----
    target = best_scope - 1
    seed_counter = 0
    while time.time() < deadline - 8:
        if target < 112:
            break
        search_time = 10.0
        if time.time() > deadline - 20:
            search_time = max(2.0, deadline - time.time() - 5)
        if search_time < 2.0:
            break

        for bw in [15, 30, 50]:
            if time.time() > deadline - 8:
                break
            res = beam_search_rows(target, bw, search_time / 3, seed=seed_counter)
            seed_counter += 1
            if res and validate(res):
                sc = get_scope(res)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = res
                    # Polish the beam search result
                    if time.time() < deadline - 10:
                        rr, sc2 = exact_local_polish(res, min(10.0, deadline - time.time() - 5), seed=seed_counter)
                        if validate(rr) and sc2 < best_scope:
                            best_scope = sc2
                            best_rows = rr
                    target = best_scope - 1
                    break
        else:
            if time.time() < deadline - 12:
                src = greedy_init(rand_prob=0.3, window=50, seed=seed_counter * 7 + 13)
                rr, sc = exact_local_polish(src, min(8.0, deadline - time.time() - 6), seed=seed_counter * 11 + 5)
                if validate(rr) and sc < best_scope:
                    best_scope = sc
                    best_rows = rr
                    target = best_scope - 1
                else:
                    break
            else:
                break
        seed_counter += 1

    # ---- Final polish with remaining time ----
    if best_rows is not None and time.time() < deadline - 3:
        final_budget = max(1.0, deadline - time.time() - 2)
        rr, sc = exact_local_polish(best_rows, final_budget, seed=99999)
        if validate(rr) and sc < best_scope:
            best_rows = rr
            best_scope = sc

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
