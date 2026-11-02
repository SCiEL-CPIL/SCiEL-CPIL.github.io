import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 143.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')

    def get_scope(rows):
        m = 0
        for r in rows:
            v = r[-1]
            if v > m:
                m = v
        return m

    def count_conflicts_fast(rows, max_val):
        seen = bytearray(max_val + 1)
        conflicts = 0
        for r in rows:
            n = len(r)
            for i in range(n):
                ri = r[i]
                for j in range(i + 1, n):
                    d = r[j] - ri
                    if seen[d]:
                        conflicts += 1
                    else:
                        seen[d] = 1
        return conflicts

    def row_diffs_list(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def all_diffs_union(rows):
        s = set()
        for r in rows:
            n = len(r)
            for i in range(n):
                ri = r[i]
                for j in range(i + 1, n):
                    s.add(r[j] - ri)
        return s

    def check_valid(rows):
        seen = set()
        for r in rows:
            n = len(r)
            for i in range(n):
                ri = r[i]
                for j in range(i + 1, n):
                    d = r[j] - ri
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_init(rand_prob=0.1, window=20, max_scope=None):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                limit = max_scope if max_scope else 400
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

    def targeted_descent(rows, time_limit):
        """Greedy descent: repeatedly reduce the largest element that can be reduced."""
        rows = [r[:] for r in rows]
        start = time.time()
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            # Find the largest element (row_idx, col_idx)
            max_val = -1
            max_r = -1
            max_c = -1
            for r_idx in range(n_rows):
                for c_idx in range(1, len(rows[r_idx])):
                    v = rows[r_idx][c_idx]
                    if v > max_val:
                        max_val = v
                        max_r = r_idx
                        max_c = c_idx
            if max_c == 0:
                break
            row = rows[max_r]
            curr = max_val
            prev = row[max_c - 1]
            # Compute fixed diffs (all diffs not involving this element)
            all_d = set()
            for r in rows:
                n = len(r)
                for i in range(n):
                    ri = r[i]
                    for j in range(i + 1, n):
                        all_d.add(r[j] - ri)
            remove = set()
            for i in range(max_c):
                remove.add(curr - row[i])
            for j in range(max_c + 1, len(row)):
                remove.add(row[j] - curr)
            fixed = all_d - remove
            # Same-row fixed (not involving c)
            same_fixed = set()
            for i in range(len(row)):
                if i == max_c:
                    continue
                for j in range(i + 1, len(row)):
                    if j == max_c:
                        continue
                    same_fixed.add(row[j] - row[i])
            eff = fixed | same_fixed

            found = False
            for nv in range(curr - 1, prev, -1):
                ok = True
                for i in range(max_c):
                    if (nv - row[i]) in eff:
                        ok = False
                        break
                if not ok:
                    continue
                for j in range(max_c + 1, len(row)):
                    if (row[j] - nv) in eff:
                        ok = False
                        break
                if not ok:
                    continue
                nd = [nv - row[i] for i in range(max_c)]
                nd += [row[j] - nv for j in range(max_c + 1, len(row))]
                if len(set(nd)) != len(nd):
                    continue
                row[max_c] = nv
                improved = True
                found = True
                break
            if not found:
                # Try all other elements in descending order
                elems = []
                for r_idx in range(n_rows):
                    for c_idx in range(1, len(rows[r_idx])):
                        elems.append((rows[r_idx][c_idx], r_idx, c_idx))
                elems.sort(reverse=True)
                for val, r_idx, c_idx in elems:
                    if (time.time() - start) > time_limit:
                        break
                    if val <= prev:
                        continue
                    row2 = rows[r_idx]
                    curr2 = row2[c_idx]
                    prev2 = row2[c_idx - 1]
                    all_d2 = set()
                    for r in rows:
                        n = len(r)
                        for i in range(n):
                            ri = r[i]
                            for j in range(i + 1, n):
                                all_d2.add(r[j] - ri)
                    remove2 = set()
                    for i in range(c_idx):
                        remove2.add(curr2 - row2[i])
                    for j in range(c_idx + 1, len(row2)):
                        remove2.add(row2[j] - curr2)
                    fixed2 = all_d2 - remove2
                    same_fixed2 = set()
                    for i in range(len(row2)):
                        if i == c_idx:
                            continue
                        for j in range(i + 1, len(row2)):
                            if j == c_idx:
                                continue
                            same_fixed2.add(row2[j] - row2[i])
                    eff2 = fixed2 | same_fixed2
                    for nv in range(curr2 - 1, prev2, -1):
                        ok = True
                        for i in range(c_idx):
                            if (nv - row2[i]) in eff2:
                                ok = False
                                break
                        if not ok:
                            continue
                        for j in range(c_idx + 1, len(row2)):
                            if (row2[j] - nv) in eff2:
                                ok = False
                                break
                        if not ok:
                            continue
                        nd = [nv - row2[i] for i in range(c_idx)]
                        nd += [row2[j] - nv for j in range(c_idx + 1, len(row2))]
                        if len(set(nd)) != len(nd):
                            continue
                        row2[c_idx] = nv
                        improved = True
                        break
                    if improved:
                        break
        return rows

    def sa_delta(rows, time_limit):
        """SA with delta conflict counting for speed."""
        current = [r[:] for r in rows]
        cur_scope = get_scope(current)
        cur_conf = count_conflicts_fast(current, cur_scope + 5)
        best = [r[:] for r in current]
        best_scope = cur_scope
        best_conf = cur_conf
        T = 12.0
        T_min = 0.01
        cooling = 0.998
        start = time.time()
        while time.time() < start + time_limit and T > T_min:
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            row = current[r_idx]
            old = row[pos]
            lo = row[pos - 1] + 1
            hi = row[pos + 1] - 1 if pos < k else cur_scope + 20
            if lo > hi:
                T *= cooling
                continue
            nv = random.randint(lo, hi)
            if nv == old:
                T *= cooling
                continue
            row[pos] = nv
            new_scope = get_scope(current)
            new_conf = count_conflicts_fast(current, new_scope + 5)
            dc = new_conf - cur_conf
            ds = new_scope - cur_scope
            accept = False
            if new_conf < cur_conf:
                accept = True
            elif new_conf == cur_conf:
                if new_scope < cur_scope:
                    accept = True
                elif random.random() < math.exp(-max(1, ds) / T):
                    accept = True
            else:
                if random.random() < math.exp(-dc / (T + 0.1)):
                    accept = True
            if accept:
                cur_conf = new_conf
                cur_scope = new_scope
                if cur_conf < best_conf or (cur_conf == best_conf and cur_scope < best_scope):
                    best_conf = cur_conf
                    best_scope = cur_scope
                    best = [r[:] for r in current]
            else:
                row[pos] = old
            T *= cooling
        return best, best_scope

    def rebuild_row(used_other, max_val=None, jitter=0.0):
        row = [0]
        row_d = set()
        while len(row) < row_len:
            found = False
            limit = max_val if max_val else 400
            for tx in range(row[-1] + 1, limit + 1):
                new = {tx - a for a in row}
                if not (new & used_other) and not (new & row_d):
                    if jitter > 0 and random.random() < jitter:
                        continue
                    row.append(tx)
                    row_d |= new
                    found = True
                    break
            if not found:
                return None
        return row

    def local_polish(rows, max_passes=5):
        rows = [list(r) for r in rows]
        improved = True
        pc = 0
        while improved and time.time() < deadline - 8 and pc < max_passes:
            improved = False
            pc += 1
            order = sorted(range(n_rows), key=lambda i: -rows[i][-1])
            for i in order:
                if time.time() > deadline - 8:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_list(rows[j])
                best_row = list(rows[i])
                best_max = rows[i][-1]
                nr = rebuild_row(used_other, max_val=best_max - 1)
                if nr is not None and nr[-1] < best_max:
                    best_row = nr
                    best_max = nr[-1]
                for att in range(6):
                    jitter = 0.1 + 0.05 * att
                    nr = rebuild_row(used_other, max_val=best_max - 1, jitter=jitter)
                    if nr is not None and nr[-1] < best_max:
                        best_row = nr
                        best_max = nr[-1]
                if best_row[-1] < rows[i][-1]:
                    rows[i] = best_row
                    improved = True
        return rows

    def kick(rows, num_kicks=1):
        rows = [list(r) for r in rows]
        idxs = random.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in idxs:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_list(rows[j])
            ok = False
            for att in range(10):
                jitter = 0.2 + 0.08 * att
                nr = rebuild_row(used_other, max_val=best_scope + 30, jitter=jitter)
                if nr is not None:
                    rows[idx] = nr
                    ok = True
                    break
            if not ok:
                nr = rebuild_row(used_other, max_val=best_scope + 40)
                if nr is not None:
                    rows[idx] = nr
        return rows

    def high_precision_polish(rows, time_limit):
        """Multiple passes of exhaustive single-element reduction."""
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        passes = 0
        while improved and (time.time() - start) < time_limit and passes < 30:
            improved = False
            passes += 1
            elems = []
            for r_idx in range(n_rows):
                for c_idx in range(1, len(rows[r_idx])):
                    elems.append((rows[r_idx][c_idx], r_idx, c_idx))
            elems.sort(reverse=True)
            for val, r_idx, c_idx in elems:
                if (time.time() - start) > time_limit:
                    break
                row = rows[r_idx]
                curr = row[c_idx]
                prev = row[c_idx - 1]
                all_d = set()
                for r in rows:
                    n = len(r)
                    for i in range(n):
                        ri = r[i]
                        for j in range(i + 1, n):
                            all_d.add(r[j] - ri)
                remove = set()
                for i in range(c_idx):
                    remove.add(curr - row[i])
                for j in range(c_idx + 1, len(row)):
                    remove.add(row[j] - curr)
                fixed = all_d - remove
                same_fixed = set()
                for i in range(len(row)):
                    if i == c_idx:
                        continue
                    for j in range(i + 1, len(row)):
                        if j == c_idx:
                            continue
                        same_fixed.add(row[j] - row[i])
                eff = fixed | same_fixed
                for nv in range(curr - 1, prev, -1):
                    ok = True
                    for i in range(c_idx):
                        if (nv - row[i]) in eff:
                            ok = False
                            break
                    if not ok:
                        continue
                    for j in range(c_idx + 1, len(row)):
                        if (row[j] - nv) in eff:
                            ok = False
                            break
                    if not ok:
                        continue
                    nd = [nv - row[i] for i in range(c_idx)]
                    nd += [row[j] - nv for j in range(c_idx + 1, len(row))]
                    if len(set(nd)) != len(nd):
                        continue
                    row[c_idx] = nv
                    improved = True
                    break
        return rows

    # === MAIN SEARCH ===

    # Phase 1: Multiple greedy restarts + targeted descent
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        rp = 0.03 + 0.03 * i
        w = 15 + 3 * i
        ms = best_scope - 1 if best_scope < float('inf') else None
        rows = greedy_init(rp, w, max_scope=ms)
        scope = get_scope(rows)
        conf = count_conflicts_fast(rows, scope + 5)

        t1_time = min(3.0, max(1.5, (deadline - time.time() - 30) / max(1, num_restarts - i)))
        t1_rows = targeted_descent(rows, t1_time)
        t1_scope = get_scope(t1_rows)
        t1_conf = count_conflicts_fast(t1_rows, t1_scope + 5)
        if t1_conf < best_conflicts or (t1_conf == best_conflicts and t1_scope < best_scope):
            best_conflicts = t1_conf
            best_scope = t1_scope
            best_rows = [r[:] for r in t1_rows]

        if time.time() < deadline - 30:
            sa_t = min(2.5, (deadline - time.time() - 30) / max(1, num_restarts - i))
            t2_rows, t2_scope = sa_delta(t1_rows, sa_t)
            t2_conf = count_conflicts_fast(t2_rows, t2_scope + 5)
            if t2_conf < best_conflicts or (t2_conf == best_conflicts and t2_scope < best_scope):
                best_conflicts = t2_conf
                best_scope = t2_scope
                best_rows = [r[:] for r in t2_rows]

        if best_conflicts == 0 and best_scope < 118:
            break

    # Phase 2: ILS
    if best_rows is not None:
        remaining = deadline - time.time() - 12
        if remaining > 15:
            il_time = remaining * 0.5
            start_il = time.time()
            while time.time() - start_il < il_time and time.time() < deadline - 12:
                cur = kick(best_rows, num_kicks=random.choice([1, 1, 2]))
                cur_conf = count_conflicts_fast(cur, get_scope(cur) + 5)
                if cur_conf > 0:
                    sa_t = min(2.0, (deadline - time.time() - 12) / 4)
                    cur, cur_scope = sa_delta(cur, sa_t)
                    cur_conf = count_conflicts_fast(cur, cur_scope + 5)
                pol = local_polish(cur, max_passes=3)
                pol_conf = count_conflicts_fast(pol, get_scope(pol) + 5)
                pol_scope = get_scope(pol)
                if pol_conf < best_conflicts or (pol_conf == best_conflicts and pol_scope < best_scope):
                    best_conflicts = pol_conf
                    best_scope = pol_scope
                    best_rows = [r[:] for r in pol]

    # Phase 3: Final targeted descent
    if best_rows is not None and time.time() < deadline - 8:
        ft = min(6.0, deadline - time.time() - 6)
        f1_rows = targeted_descent(best_rows, ft)
        f1_scope = get_scope(f1_rows)
        f1_conf = count_conflicts_fast(f1_rows, f1_scope + 5)
        if f1_conf < best_conflicts or (f1_conf == best_conflicts and f1_scope < best_scope):
            best_conflicts = f1_conf
            best_scope = f1_scope
            best_rows = [r[:] for r in f1_rows]

    # Phase 4: High precision polish
    if best_rows is not None and time.time() < deadline - 4:
        pt = min(15.0, deadline - time.time() - 2)
        polished = high_precision_polish(best_rows, pt)
        pol_conf = count_conflicts_fast(polished, get_scope(polished) + 5)
        if pol_conf == 0:
            pol_scope = get_scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = [r[:] for r in polished]

    # Fallback
    if best_rows is None:
        best_rows = greedy_init(0.1, 20)

    # Final safety
    if best_conflicts > 0 and time.time() < deadline - 2:
        fr, fs = sa_delta(best_rows, deadline - time.time() - 1)
        fc = count_conflicts_fast(fr, fs + 5)
        if fc < best_conflicts or (fc == best_conflicts and fs < best_scope):
            best_conflicts = fc
            best_scope = fs
            best_rows = [r[:] for r in fr]

    return best_rows
