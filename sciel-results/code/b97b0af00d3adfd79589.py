import time, random, math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 114.0

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2  # pairwise differences per row of length k+1

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def sum_of(rows):
        return sum(sum(r) for r in rows)

    def is_valid(rows):
        used = set()
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            d = row_diffs(r)
            if len(d) != needed:
                return False
            if d & used:
                return False
            used |= d
        return True

    # ---- deterministic greedy (baseline, always valid) ----
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # ---- randomized greedy: diversify the myopic "smallest" choice ----
    def rand_greedy(rng, bias, window):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                cands = []
                y = x + 1
                scanned = 0
                while len(cands) < window and scanned < 2000:
                    scanned += 1
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                while not cands:
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                if rng.random() < bias:
                    chosen = cands[0]
                else:
                    idx = int(len(cands) * (rng.random() ** 1.6))
                    idx = min(idx, len(cands) - 1)
                    chosen = cands[idx]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
            rows.append(row)
        return rows

    # ---- simulated-annealing local search that compresses the global max ----
    def local_search(start, rng, steps, T0=3.0):
        cur = [list(r) for r in start]
        rowd = [row_diffs(r) for r in cur]
        cur_used = set()
        for d in rowd:
            cur_used |= d

        # Precompute row sums and maxes for incremental obj update
        row_sums = [sum(r) for r in cur]
        row_maxes = [max(r) for r in cur]
        cur_sum = sum(row_sums)
        cur_scope = max(row_maxes)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0

        for it in range(steps):
            if it % 500 == 0 and remaining() < 12:
                break
            r = rng.randrange(n_rows)
            row = cur[r]

            # Move generation
            # 55%: Attack the maximum (scope driver)
            # 25%: Attack a random interior point
            # 20%: "Step" move on the maximum (try to decrement by 1)
            move_type = rng.random()
            if move_type < 0.55:
                p = row.index(max(row))
                if p == 0:
                    continue
                lo = row[p - 1] + 1
                hi = row[p] - 1
            elif move_type < 0.80:
                p = rng.randrange(1, len(row))
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 30)
            else:
                # Step move on max
                p = row.index(max(row))
                if p == 0:
                    continue
                v = row[p] - 1
                if v <= row[p-1]:
                    continue
                # Handle this case separately to avoid lo/hi logic
                new_row = row[:]
                new_row[p] = v
                if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                other = cur_used - rowd[r]
                if nd & other:
                    continue
                # Calculate new obj
                new_max_row = max(new_row)
                new_sum_row = sum(new_row)

                # Update global scope
                if new_max_row < row_maxes[r]:
                    # Max decreased, need to check if it's still the global max
                    if row_maxes[r] == cur_scope:
                         # Potential drop in scope
                         # Find max of other rows
                         others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                         new_scope = max(new_max_row, others_max)
                    else:
                         new_scope = cur_scope
                else:
                    new_scope = max(cur_scope, new_max_row)

                new_sum = cur_sum - row_sums[r] + new_sum_row
                new_obj = new_scope * 1000000 + new_sum
                d = new_obj - cur_obj

                if d <= 0 or rng.random() < math.exp(-d / T):
                    cur[r] = new_row
                    rowd[r] = nd
                    cur_used = other | nd
                    row_sums[r] = new_sum_row
                    row_maxes[r] = new_max_row
                    cur_sum = new_sum
                    cur_scope = new_scope
                    cur_obj = new_obj
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best = [list(x) for x in cur]
                T = max(0.05, T * 0.9995)
                continue

            if lo > hi:
                continue
            v = rng.randrange(lo, hi + 1)
            if v == row[p]:
                continue

            new_row = row[:]
            new_row[p] = v
            if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                continue

            nd = row_diffs(new_row)
            if len(nd) != needed:          # must stay a Golomb ruler internally
                continue

            other = cur_used - rowd[r]
            if nd & other:                 # no cross-row collision
                continue

            # Incremental Objective Update
            new_max_row = max(new_row)
            new_sum_row = sum(new_row)

            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)

            new_sum = cur_sum - row_sums[r] + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj

            if d <= 0 or rng.random() < math.exp(-d / T):
                cur[r] = new_row
                rowd[r] = nd
                cur_used = other | nd
                row_sums[r] = new_sum_row
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]

            T = max(0.05, T * 0.9995)
        return best

    # ---- main search with explicit deadline guard ----
    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    rng = random.Random(20240607)

    # Phase 1: many fast randomized-greedy restarts
    while remaining() > 30:
        bias = rng.uniform(0.55, 0.97)
        window = rng.choice([3, 4, 5, 6, 8, 10])
        rows = rand_greedy(rng, bias, window)
        s = scope_of(rows)
        if s < best_scope:
            best_scope = s
            best_rows = rows
        # quick polish of promising candidates
        if s < best_scope + 6 and remaining() > 18:
            pol = local_search(rows, rng, 400)
            sp = scope_of(pol)
            if sp < best_scope:
                best_scope = sp
                best_rows = pol

    # Phase 2: deeper annealing from the best + perturbed restarts
    for _ in range(8):
        if remaining() < 8:
            break
        res = local_search(best_rows, rng, 3500, T0=rng.uniform(2.0, 6.0))
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res
        else:
            # perturb the best to escape its basin, then re-anneal
            pert = [list(r) for r in best_rows]
            for _ in range(4):
                rr = rng.randrange(n_rows)
                pp = rng.randrange(1, len(pert[rr]))
                lo = pert[rr][pp - 1] + 1
                hi = (pert[rr][pp + 1] - 1) if pp + 1 < len(pert[rr]) else best_scope + 25
                if lo > hi:
                    continue
                nv = rng.randrange(lo, hi + 1)
                new_row = pert[rr][:]
                new_row[pp] = nv
                if not all(new_row[i] < new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                used, ok = set(), True
                for i, row in enumerate(pert):
                    d = nd if i == rr else row_diffs(row)
                    if len(d) != needed or (d & used):
                        ok = False
                        break
                    used |= d
                if ok:
                    pert[rr] = new_row
                    break
            res = local_search(pert, rng, 2500, T0=4.0)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    # Phase 3: final intensive polish if time remains
    if remaining() > 5:
        res = local_search(best_rows, rng, 3000, T0=2.0)
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res

    # Final safety: guarantee a valid result
    if not is_valid(best_rows):
        best_rows = base
    return best_rows
