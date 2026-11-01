import time
import random
import math
import heapq

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

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

    # ---- Greedy initialization (fast, diverse) ----
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

    # ---- Local search with targeted descent + row reconstruction ----
    def local_search(rows, time_limit, seed=None):
        rng = random.Random(seed)
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)

        def get_all_diffs(rows):
            diffs = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs

        # Try to reduce global max via single-element moves; if stuck, reconstruct a row.
        no_improve_count = 0
        while time.time() < start + time_limit:
            # Find global max element
            max_val = -1
            max_r = max_c = -1
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r = r_idx
                        max_c = c_idx
            if max_c == 0:
                break

            row = best_rows[max_r]
            prev_val = row[max_c - 1]
            all_diffs = get_all_diffs(best_rows)
            remove_set = set()
            for i in range(max_c):
                remove_set.add(max_val - row[i])
            for j in range(max_c + 1, len(row)):
                remove_set.add(row[j] - max_val)
            fixed_diffs = all_diffs - remove_set

            found = False
            for new_val in range(max_val - 1, prev_val, -1):
                ok = True
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
                nd = [new_val - row[i] for i in range(max_c)] + [row[j] - new_val for j in range(max_c + 1, len(row))]
                if len(set(nd)) != len(nd):
                    continue
                row[max_c] = new_val
                found = True
                break

            if found:
                ns = get_scope(best_rows)
                if ns < best_scope:
                    best_scope = ns
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Perturb: reconstruct a random row entirely (structural escape)
                r_idx = rng.randint(0, n_rows - 1)
                # Remove this row's diffs
                all_diffs = get_all_diffs(best_rows)
                row = best_rows[r_idx]
                remove_set = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        remove_set.add(row[j] - row[i])
                fixed_diffs = all_diffs - remove_set

                # Rebuild row greedily against fixed_diffs, with randomization
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
                    # try a quick descent pass
                    changed = True
                    while changed and time.time() < start + time_limit:
                        changed = False
                        mv = -1
                        mr = mc = -1
                        for ri, rr in enumerate(best_rows):
                            for ci, vv in enumerate(rr):
                                if vv > mv:
                                    mv = vv
                                    mr, mc = ri, ci
                        if mc == 0:
                            break
                        rr = best_rows[mr]
                        pv = rr[mc - 1]
                        ad = get_all_diffs(best_rows)
                        rm = set()
                        for i in range(mc):
                            rm.add(mv - rr[i])
                        for j in range(mc + 1, len(rr)):
                            rm.add(rr[j] - mv)
                        fd = ad - rm
                        for nv in range(mv - 1, pv, -1):
                            ok = True
                            for i in range(mc):
                                if (nv - rr[i]) in fd:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            for j in range(mc + 1, len(rr)):
                                if (rr[j] - nv) in fd:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            nd = [nv - rr[i] for i in range(mc)] + [rr[j] - nv for j in range(mc + 1, len(rr))]
                            if len(set(nd)) != len(nd):
                                continue
                            rr[mc] = nv
                            changed = True
                            break
                    ns = get_scope(best_rows)
                    if ns < best_scope:
                        best_scope = ns
                    no_improve_count = 0

        return best_rows, best_scope

    # ---- Best-first search over difference slots (A* with admissible heuristic) ----
    # State: (row_idx, pos, used_diffs, rows, max_val)
    # Heuristic: lower bound on final max from remaining required distinct diffs.
    # We use a simple admissible heuristic: the remaining number of distinct diffs
    # that must be placed, each needing a distinct integer in (0, max], so max >= count.
    # Combined with current max, gives max(current_max, total_diffs_placed + remaining).

    def best_first_search(target_scope, time_limit, seed=0):
        rng = random.Random(seed)
        start = time.time()

        rows = [[0] * row_len for _ in range(n_rows)]
        used = set()
        # We'll do depth-first-ish with memoization of dead states, but guided by best-first.
        # For tractability, use recursive DFS with ordering by value and a node budget,
        # but restart with different random orderings (this is the "best-first" flavor:
        # we try the most promising (smallest) values first, and the heuristic prunes).

        nodes = [0]

        def dfs(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            nodes[0] += 1
            if row_idx == n_rows:
                return True
            if pos == row_len:
                return dfs(row_idx + 1, 1)

            prev_val = rows[row_idx][pos - 1]
            lower = prev_val + 1
            upper = target_scope

            # Admissible heuristic: we still need (n_rows - row_idx - 1) full rows + (row_len - pos - 1) more
            # entries in current row. Each new entry adds (pos+1) diffs. The total distinct diffs so far
            # is len(used). Final max must be >= total distinct diffs (since diffs are 1..max distinct).
            remaining_entries = (row_len - pos - 1) + (n_rows - row_idx - 1) * row_len
            # each remaining entry adds at least 1 diff, so final distinct count >= len(used) + remaining_entries
            if len(used) + remaining_entries > target_scope:
                return False

            # Try values; order: mostly ascending (smallest first = best-first), with occasional shuffle
            vals = list(range(lower, upper + 1))
            if len(vals) > 60:
                # limit candidates to a window around lower + some random spread
                step = max(1, len(vals) // 60)
                vals = vals[::step]
                rng.shuffle(vals[10:])  # keep small values ordered, shuffle the rest

            for val in vals:
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used:
                        valid = False
                        break
                if not valid:
                    continue
                new_diffs = [val - rows[row_idx][i] for i in range(pos)]
                for d in new_diffs:
                    used.add(d)
                rows[row_idx][pos] = val
                if dfs(row_idx, pos + 1):
                    return True
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used.remove(d)
            return False

        if dfs(0, 1):
            return [r[:] for r in rows]
        return None

    # ---- Phase 1: diverse greedy inits ----
    best_rows = None
    best_scope = float('inf')
    num_inits = 12
    init_budget = 6.0
    t0 = time.time()
    for i in range(num_inits):
        if time.time() > t0 + init_budget:
            break
        rows = greedy_init(rand_prob=0.05 + 0.01 * i, window=30, seed=i * 1000 + 7)
        sc = get_scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = [r[:] for r in rows]

    # ---- Phase 2: local search refinement (multiple seeds, keep best) ----
    ls_budget = 50.0
    if time.time() > deadline - 60:
        ls_budget = max(5.0, deadline - time.time() - 10)
    if best_rows is not None and ls_budget > 5:
        # Run local search from a few different starting points
        best_ls = best_rows
        best_ls_scope = best_scope
        per = max(5.0, ls_budget / 3)
        for s in range(3):
            if time.time() > deadline - 10:
                break
            # Start from best or a fresh greedy
            if s == 0:
                src = best_rows
            else:
                src = greedy_init(rand_prob=0.2, window=40, seed=5000 + s * 31)
            rr, sc = local_search(src, per, seed=9000 + s * 17)
            if validate(rr) and sc < best_ls_scope:
                best_ls = rr
                best_ls_scope = sc
        if validate(best_ls):
            best_rows = best_ls
            best_scope = best_ls_scope

    # ---- Phase 3: best-first / DFS to try to beat current scope ----
    target = best_scope - 1
    seed_counter = 0
    while time.time() < deadline - 6:
        if target < 112:
            break
        search_time = 6.0
        if time.time() > deadline - 16:
            search_time = max(1.0, deadline - time.time() - 3)
        if search_time < 1.0:
            break
        res = best_first_search(target, search_time, seed=seed_counter)
        seed_counter += 1
        if res and validate(res):
            sc = get_scope(res)
            if sc < best_scope:
                best_scope = sc
                best_rows = res
                target = best_scope - 1
        else:
            # one retry with different seed if time allows
            if time.time() < deadline - 10:
                res2 = best_first_search(target, 5.0, seed=seed_counter)
                seed_counter += 1
                if res2 and validate(res2):
                    sc = get_scope(res2)
                    if sc < best_scope:
                        best_scope = sc
                        best_rows = res2
                        target = best_scope - 1
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
