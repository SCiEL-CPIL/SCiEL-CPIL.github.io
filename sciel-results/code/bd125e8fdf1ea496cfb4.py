import time
import random
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0  # safety margin before 150s kill

    d = k * (k + 1) // 2
    rng = random.Random(12345)

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        seen = set()
        scope = 0
        for row in rows:
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, row[-1])
        return scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def build_diff_counts(rows):
        dc = {}
        for r in rows:
            n = len(r)
            for i in range(n):
                ri = r[i]
                for j in range(i + 1, n):
                    dd = r[j] - ri
                    dc[dd] = dc.get(dd, 0) + 1
        return dc

    def conflict_count_from(dc):
        c = 0
        for v in dc.values():
            if v > 1:
                c += 1
        return c

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = full_validate_scope(best_rows)
    if best_scope is None:
        best_scope = get_scope(best_rows)

    # ---- Targeted descent: shrink the max mark of the row(s) holding the global max ----
    def targeted_descent(rows, budget):
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            return None, None
        improved = False
        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [i for i in range(n_rows) if cur[i][-1] == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if old_val <= lower:
                continue
            candidates = set()
            for v in range(lower, min(lower + 20, old_val)):
                candidates.add(v)
            for _ in range(12):
                candidates.add(rng.randint(lower, old_val - 1))
            found = False
            for new_val in sorted(candidates, key=lambda x: -x):
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                scope = full_validate_scope(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        found = True
                        break
                row[pos] = old_val
            if found:
                continue
            # try a non-last element of the max row
            if len(row) > 2:
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 > lo2:
                    for _ in range(6):
                        nv = rng.randint(lo2, hi2)
                        if nv == old2:
                            continue
                        row[pos2] = nv
                        scope = full_validate_scope(cur)
                        if scope is not None and scope < cur_scope:
                            cur_scope = scope
                            improved = True
                            found = True
                            break
                        row[pos2] = old2
                    if found:
                        continue
            break
        if improved:
            return cur, cur_scope
        return None, None

    # ---- CDVR: find the most-colliding row-position, swap to smallest valid alternative ----
    def cdvr_pass(rows, dc, budget):
        start = time.time()
        cur = [r[:] for r in rows]
        cur_dc = {kk: v for kk, v in dc.items()}
        cur_conf = conflict_count_from(cur_dc)
        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            return None, None, None
        improved = False
        while time.time() - start < budget:
            # score each (row,pos) by how many of its differences are in conflict
            best_score = -1
            best_rp = None
            for ri in range(n_rows):
                row = cur[ri]
                n = len(row)
                for pos in range(1, n):
                    sc = 0
                    for i in range(pos):
                        dd = row[pos] - row[i]
                        if cur_dc.get(dd, 0) > 1:
                            sc += 1
                    for j in range(pos + 1, n):
                        dd = row[j] - row[pos]
                        if cur_dc.get(dd, 0) > 1:
                            sc += 1
                    if sc > best_score:
                        best_score = sc
                        best_rp = (ri, pos)
            if best_score <= 0:
                break
            ri, pos = best_rp
            row = cur[ri]
            old_val = row[pos]
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1 if pos < len(row) - 1 else cur_scope + 30
            if hi <= lo:
                break
            # candidate pool: smallest values first (prefer shrink), then some random
            cands = []
            for v in range(lo, min(lo + 25, hi + 1)):
                if v != old_val:
                    cands.append(v)
            for _ in range(15):
                v = rng.randint(lo, hi)
                if v != old_val:
                    cands.append(v)
            placed = False
            for new_val in cands:
                # compute removed/added diffs
                rem = []
                add = []
                for i in range(pos):
                    rem.append(old_val - row[i])
                    add.append(new_val - row[i])
                for j in range(pos + 1, len(row)):
                    rem.append(row[j] - old_val)
                    add.append(row[j] - new_val)
                # simulate
                tmp_conf = cur_conf
                for dd in rem:
                    oc = cur_dc.get(dd, 0)
                    nc = oc - 1
                    if nc == 0:
                        del cur_dc[dd]
                    else:
                        cur_dc[dd] = nc
                    if oc == 2 and nc == 1:
                        tmp_conf -= 1
                for dd in add:
                    oc = cur_dc.get(dd, 0)
                    nc = oc + 1
                    cur_dc[dd] = nc
                    if oc == 1 and nc == 2:
                        tmp_conf += 1
                row[pos] = new_val
                ns = full_validate_scope(cur)
                if ns is not None and tmp_conf == 0:
                    # valid and conflict-free
                    cur_scope = ns
                    cur_conf = 0
                    improved = True
                    placed = True
                    break
                else:
                    # rollback
                    row[pos] = old_val
                    for dd in add:
                        c = cur_dc[dd]
                        if c == 1:
                            del cur_dc[dd]
                        else:
                            cur_dc[dd] = c - 1
                        if c == 2:
                            tmp_conf -= 1
                    for dd in rem:
                        c = cur_dc.get(dd, 0)
                        cur_dc[dd] = c + 1
                        if c == 1:
                            tmp_conf += 1
                    cur_conf = tmp_conf
            if not placed:
                break
        if improved:
            return cur, cur_dc, cur_scope
        return None, None, None

    # ---- Fast incremental simulated annealing ----
    def anneal(cur, cur_dc, cur_conf, cur_scope, T0, steps, budget):
        start = time.time()
        T = T0
        no_improve = 0
        local_best = (cur_scope, [r[:] for r in cur])
        for _ in range(steps):
            if time.time() > budget:
                break
            ri = rng.randrange(n_rows)
            row = cur[ri]
            n = len(row)
            pos = rng.randint(1, n - 1)
            old_val = row[pos]
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1 if pos < n - 1 else cur_scope + 40
            if hi <= lo:
                continue
            # bias: prefer smaller values to shrink scope
            if rng.random() < 0.55 and old_val > lo:
                new_val = rng.randint(lo, old_val)
            else:
                new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            rem = []
            add = []
            for i in range(pos):
                rem.append(old_val - row[i])
                add.append(new_val - row[i])
            for j in range(pos + 1, n):
                rem.append(row[j] - old_val)
                add.append(row[j] - new_val)
            tmp_conf = cur_conf
            for dd in rem:
                oc = cur_dc.get(dd, 0)
                nc = oc - 1
                if nc == 0:
                    del cur_dc[dd]
                else:
                    cur_dc[dd] = nc
                if oc == 2 and nc == 1:
                    tmp_conf -= 1
            for dd in add:
                oc = cur_dc.get(dd, 0)
                nc = oc + 1
                cur_dc[dd] = nc
                if oc == 1 and nc == 2:
                    tmp_conf += 1
            row[pos] = new_val
            # new scope
            new_row_max = row[-1]
            ns = 0
            for idx, r in enumerate(cur):
                m = new_row_max if idx == ri else r[-1]
                if m > ns:
                    ns = m
            # cost: conflicts dominate, then scope
            cost = tmp_conf * 100000 + ns
            old_cost = cur_conf * 100000 + cur_scope
            delta = cost - old_cost
            if delta <= 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
                cur_conf = tmp_conf
                cur_scope = ns
                if tmp_conf == 0 and ns < local_best[0]:
                    local_best = (ns, [r[:] for r in cur])
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve > 4000:
                    # kick
                    for _ in range(4):
                        kri = rng.randrange(n_rows)
                        krow = cur[kri]
                        kn = len(krow)
                        kpos = rng.randint(1, kn - 1)
                        klo = krow[kpos - 1] + 1
                        khi = krow[kpos + 1] - 1 if kpos < kn - 1 else cur_scope + 20
                        if khi > klo:
                            kv = krow[kpos]
                            for i in range(kpos):
                                dd = kv - krow[i]
                                oc = cur_dc.get(dd, 0); nc = oc - 1
                                if nc == 0: del cur_dc[dd]
                                else: cur_dc[dd] = nc
                                if oc == 2 and nc == 1: cur_conf -= 1
                            nnv = rng.randint(klo, khi)
                            krow[kpos] = nnv
                            for i in range(kpos):
                                dd = nnv - krow[i]
                                oc = cur_dc.get(dd, 0); nc = oc + 1
                                cur_dc[dd] = nc
                                if oc == 1 and nc == 2: cur_conf += 1
                            for j in range(kpos + 1, kn):
                                dd = krow[j] - nnv
                                oc = cur_dc.get(dd, 0); nc = oc + 1
                                cur_dc[dd] = nc
                                if oc == 1 and nc == 2: cur_conf += 1
                            for i in range(kpos):
                                dd = nnv - krow[i]
                                oc = cur_dc.get(dd, 0); nc = oc - 1
                                if nc == 0: del cur_dc[dd]
                                else: cur_dc[dd] = nc
                                if oc == 2 and nc == 1: cur_conf -= 1
                            for j in range(kpos + 1, kn):
                                dd = krow[j] - nnv
                                oc = cur_dc.get(dd, 0); nc = oc - 1
                                if nc == 0: del cur_dc[dd]
                                else: cur_dc[dd] = nc
                                if oc == 2 and nc == 1: cur_conf -= 1
                    no_improve = 0
                    T = max(T, 30.0)
            else:
                # rollback
                row[pos] = old_val
                for dd in add:
                    c = cur_dc[dd]
                    if c == 1:
                        del cur_dc[dd]
                    else:
                        cur_dc[dd] = c - 1
                    if c == 2:
                        tmp_conf -= 1
                for dd in rem:
                    c = cur_dc.get(dd, 0)
                    cur_dc[dd] = c + 1
                    if c == 1:
                        tmp_conf += 1
                cur_conf = tmp_conf
                no_improve += 1
            T *= 0.99995
        return cur, cur_dc, cur_conf, cur_scope, local_best

    # Main loop over restarts
    max_restarts = 400
    for restart in range(max_restarts):
        if time.time() > deadline - 3.0:
            break
        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 15
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)
        cur_dc = build_diff_counts(cur)
        cur_conf = conflict_count_from(cur_dc)
        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            # invalid; try to repair via CDVR
            cd = cdvr_pass(cur, cur_dc, budget=2.0)
            if cd[0] is None:
                continue
            cur, cur_dc, cur_scope = cd
        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # anneal with a budget
        budget = min(deadline - 3.0, time.time() + 3.0)
        T0 = max(0.05, 1.0 - restart / max_restarts)
        cur, cur_dc, cur_conf, cur_scope, local_best = anneal(
            cur, cur_dc, cur_conf, cur_scope, T0, 40000, budget)
        if local_best[0] < best_scope:
            best_scope = local_best[0]
            best_rows = local_best[1]

        # CDVR polish from the annealed state
        if time.time() < deadline - 5.0:
            cd = cdvr_pass(cur, cur_dc, budget=2.0)
            if cd[0] is not None and cd[2] is not None and cd[2] < best_scope:
                best_scope = cd[2]
                best_rows = cd[0]

        # targeted descent
        if time.time() < deadline - 4.0:
            td = targeted_descent(best_rows, budget=1.5)
            if td[0] is not None and td[1] < best_scope:
                best_scope = td[1]
                best_rows = td[0]

    # Final greedy scope-shrink polish on the best
    if time.time() < deadline - 1.5:
        cur = [r[:] for r in best_rows]
        cur_dc = build_diff_counts(cur)
        cur_conf = conflict_count_from(cur_dc)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for ri in range(n_rows):
                row = cur[ri]
                for pos in range(len(row) - 1, 0, -1):
                    old_val = row[pos]
                    lo = row[pos - 1] + 1
                    if lo >= old_val:
                        continue
                    for new_val in range(old_val - 1, lo - 1, -1):
                        rem = []
                        add = []
                        for i in range(pos):
                            rem.append(old_val - row[i])
                            add.append(new_val - row[i])
                        for j in range(pos + 1, len(row)):
                            rem.append(row[j] - old_val)
                            add.append(row[j] - new_val)
                        tmp_conf = cur_conf
                        for dd in rem:
                            oc = cur_dc.get(dd, 0); nc = oc - 1
                            if nc == 0: del cur_dc[dd]
                            else: cur_dc[dd] = nc
                            if oc == 2 and nc == 1: tmp_conf -= 1
                        for dd in add:
                            oc = cur_dc.get(dd, 0); nc = oc + 1
                            cur_dc[dd] = nc
                            if oc == 1 and nc == 2: tmp_conf += 1
                        row[pos] = new_val
                        ns = full_validate_scope(cur)
                        if ns is not None and tmp_conf == 0:
                            cur_conf = 0
                            if ns < best_scope:
                                best_scope = ns
                                best_rows = [r[:] for r in cur]
                            improved = True
                            break
                        else:
                            row[pos] = old_val
                            for dd in add:
                                c = cur_dc[dd]
                                if c == 1: del cur_dc[dd]
                                else: cur_dc[dd] = c - 1
                                if c == 2: tmp_conf -= 1
                            for dd in rem:
                                c = cur_dc.get(dd, 0)
                                cur_dc[dd] = c + 1
                                if c == 1: tmp_conf += 1
                            cur_conf = tmp_conf
                            break

    # safety: ensure validity of returned best
    if full_validate_scope(best_rows) is None:
        # rebuild from greedy
        best_rows = greedy_init()
        best_scope = full_validate_scope(best_rows) or get_scope(best_rows)
    return best_rows


if __name__ == "__main__":
    rows = construct(4, 7)
    print(rows)
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
            break
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
        if not valid:
            break
    print("Valid:", valid)
    print("Scope:", max(max(r) for r in rows))
