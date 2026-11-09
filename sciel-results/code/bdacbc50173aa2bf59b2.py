import time, random, math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 140.0

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2

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

    def is_valid(rows):
        used = set()
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i+1] for i in range(len(r)-1)):
                return False
            d = row_diffs(r)
            if len(d) != needed:
                return False
            if d & used:
                return False
            used |= d
        return True

    # Greedy baseline
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

    # Randomized greedy
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
                while len(cands) < window and scanned < 3000:
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
                    idx = int(len(cands) * (rng.random() ** 1.5))
                    idx = min(idx, len(cands) - 1)
                    chosen = cands[idx]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
            rows.append(row)
        return rows

    # Row rebuild: rebuild row r from scratch given fixed constraints of others
    def rebuild_row(rows, r, rng, time_budget=0.5):
        others_used = set()
        for i in range(n_rows):
            if i != r:
                others_used |= row_diffs(rows[i])
        scope = max(max(row) for row in rows)

        # Try multiple randomized greedy builds, keep best scope
        best_row = None
        best_scope = scope + 50
        t_start = time.time()
        attempts = 0
        while time.time() - t_start < time_budget and attempts < 200:
            attempts += 1
            row = [0]
            x = 0
            while len(row) < k + 1:
                cands = []
                y = x + 1
                scanned = 0
                while len(cands) < 5 and scanned < 500:
                    scanned += 1
                    new = {y - a for a in row}
                    if not (new & others_used):
                        cands.append(y)
                    y += 1
                while not cands:
                    new = {y - a for a in row}
                    if not (new & others_used):
                        cands.append(y)
                    y += 1
                # Randomized choice
                if rng.random() < 0.7:
                    chosen = cands[0]
                else:
                    idx = int(len(cands) * (rng.random() ** 1.3))
                    idx = min(idx, len(cands) - 1)
                    chosen = cands[idx]
                row.append(chosen)
                x = chosen
            row_scope = max(row)
            if row_scope < best_scope:
                best_scope = row_scope
                best_row = row[:]
        return best_row

    # Local search with single-element moves + row rebuild
    def local_search(start, rng, steps, T0=3.0, allow_rebuild=True):
        cur = [list(r) for r in start]
        rowd = [row_diffs(r) for r in cur]
        cur_used = set()
        for d in rowd:
            cur_used |= d

        row_sums = [sum(r) for r in cur]
        row_maxes = [max(r) for r in cur]
        cur_sum = sum(row_sums)
        cur_scope = max(row_maxes)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0
        rebuild_count = 0

        for it in range(steps):
            if it % 300 == 0 and remaining() < 15:
                break
            r = rng.randrange(n_rows)
            row = cur[r]

            # Decide move type
            move_type = rng.random()
            if allow_rebuild and move_type < 0.20 and rebuild_count < 30:
                # Row rebuild
                new_row = rebuild_row(cur, r, rng, time_budget=0.3)
                if new_row is None:
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                other = cur_used - rowd[r]
                if nd & other:
                    continue
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
                d_obj = new_obj - cur_obj
                if d_obj <= 0 or rng.random() < math.exp(-d_obj / T):
                    cur[r] = new_row
                    rowd[r] = nd
                    cur_used = other | nd
                    row_sums[r] = new_sum_row
                    row_maxes[r] = new_max_row
                    cur_sum = new_sum
                    cur_scope = new_scope
                    cur_obj = new_obj
                    rebuild_count += 1
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best = [list(x) for x in cur]
                T = max(0.05, T * 0.999)
                continue

            # Single-element move
            if move_type < 0.55:
                p = row.index(max(row))
                if p == 0:
                    continue
                lo = row[p-1] + 1
                hi = row[p] - 1
            elif move_type < 0.85:
                p = rng.randrange(1, len(row))
                lo = row[p-1] + 1
                hi = (row[p+1] - 1) if p + 1 < len(row) else (cur_scope + 40)
            else:
                # Step move on max
                p = row.index(max(row))
                if p == 0:
                    continue
                v = row[p] - 1
                if v <= row[p-1]:
                    continue
                new_row = row[:]
                new_row[p] = v
                if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                other = cur_used - rowd[r]
                if nd & other:
                    continue
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
                d_obj = new_obj - cur_obj
                if d_obj <= 0 or rng.random() < math.exp(-d_obj / T):
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
                T = max(0.05, T * 0.999)
                continue

            if lo > hi:
                continue
            v = rng.randrange(lo, hi + 1)
            if v == row[p]:
                continue
            new_row = row[:]
            new_row[p] = v
            if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                continue
            nd = row_diffs(new_row)
            if len(nd) != needed:
                continue
            other = cur_used - rowd[r]
            if nd & other:
                continue
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
            d_obj = new_obj - cur_obj
            if d_obj <= 0 or rng.random() < math.exp(-d_obj / T):
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
            T = max(0.05, T * 0.999)
        return best

    # Main search
    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    rng = random.Random(42)

    # Phase 1: randomized greedy restarts
    while remaining() > 35:
        bias = rng.uniform(0.5, 0.95)
        window = rng.choice([3, 4, 5, 6, 8, 10, 12])
        rows = rand_greedy(rng, bias, window)
        s = scope_of(rows)
        if s < best_scope:
            best_scope = s
            best_rows = rows
        if s < best_scope + 8 and remaining() > 20:
            pol = local_search(rows, rng, 500, T0=2.5)
            sp = scope_of(pol)
            if sp < best_scope:
                best_scope = sp
                best_rows = pol

    # Phase 2: deeper annealing with rebuilds
    for _ in range(10):
        if remaining() < 10:
            break
        res = local_search(best_rows, rng, 4000, T0=rng.uniform(2.0, 5.0), allow_rebuild=True)
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res
        else:
            # Perturb and re-anneal
            pert = [list(r) for r in best_rows]
            for _ in range(6):
                rr = rng.randrange(n_rows)
                pp = rng.randrange(1, len(pert[rr]))
                lo = pert[rr][pp-1] + 1
                hi = (pert[rr][pp+1] - 1) if pp + 1 < len(pert[rr]) else best_scope + 30
                if lo > hi:
                    continue
                nv = rng.randrange(lo, hi + 1)
                new_row = pert[rr][:]
                new_row[pp] = nv
                if not all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)):
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
            res = local_search(pert, rng, 3000, T0=4.0, allow_rebuild=True)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    # Phase 3: final polish
    if remaining() > 6:
        res = local_search(best_rows, rng, 3000, T0=2.0, allow_rebuild=True)
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res

    if not is_valid(best_rows):
        best_rows = base
    return best_rows
