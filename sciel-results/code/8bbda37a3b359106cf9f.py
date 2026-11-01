import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def _time_left():
        return deadline - time.time()

    # ---------- validation ----------
    def _validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def _row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def _global_max(rows):
        return max(max(r) for r in rows)

    # ---------- greedy initialization ----------
    def greedy_construct(n_rows, k, rng=None):
        if rng is None:
            rng = random
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = _global_max(best_solution)

    # ---------- randomized greedy pool ----------
    def randomized_greedy(rng):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # ---------- candidate pool of single valid Sidon rows ----------
    def _is_sidon_row(row):
        d = _row_diffs(row)
        return len(d) == len(row) * (len(row) - 1) // 2

    def build_pool(n_rows, k, scope_limit, target_size, rng, start_time, deadline):
        pool = {}
        # add greedy rows
        for _ in range(3):
            rows = randomized_greedy(rng)
            for r in rows:
                if max(r) <= scope_limit and _is_sidon_row(r):
                    pool[tuple(r)] = _row_diffs(r)
        # add single-row random constructions bounded by scope_limit
        attempts = 0
        max_attempts = 40000
        while len(pool) < target_size and attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            # random increasing set of k+1 within [0, scope_limit]
            if scope_limit < k + 1:
                break
            row = [0]
            prev = 0
            for _ in range(k):
                hi = scope_limit
                # ensure room for remaining
                remaining = k - len(row) + 1
                if hi - prev < remaining:
                    break
                # bias toward larger values to fill scope
                v = prev + 1 + rng.randint(0, max(0, hi - prev - remaining))
                row.append(v)
                prev = v
            if len(row) != k + 1:
                continue
            if not _is_sidon_row(row):
                continue
            t = tuple(row)
            if t not in pool:
                pool[t] = _row_diffs(row)
        return pool

    # ---------- local search: move one element in one row ----------
    def local_search_move(rows, all_used, row_diffs, row_maxes, n_rows, k, rng,
                          start_time, deadline, iters):
        current_rows = [r[:] for r in rows]
        current_val = max(row_maxes)
        best_rows = [r[:] for r in current_rows]
        best_val = current_val

        def try_move(ri, pi, new_val):
            nonlocal current_val, best_val, best_rows
            if new_val <= current_rows[ri][pi - 1]:
                return False
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                return False
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            new_diffs = _row_diffs(cand_row)
            if len(new_diffs) != len(cand_row) * (len(cand_row) - 1) // 2:
                return False
            # cross-row conflicts
            for d in new_diffs:
                if d in all_used and d not in row_diffs[ri]:
                    return False
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global = max(row_maxes)
            row_maxes[ri] = old_row_max
            delta = new_global - current_val
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.15:
                    accept = True
            else:
                if rng.random() < 0.02:
                    accept = True
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                return True
            return False

        for _ in range(iters):
            if time.time() > deadline - 5:
                break
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            old = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            # try decreasing first (attack the max)
            step = max(1, (old - min_allowed) // 12) if old > min_allowed else 1
            for d in range(1, min(old - min_allowed + 1, 40)):
                if try_move(ri, pi, old - d):
                    break
            else:
                # exploration
                hi = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
                if min_allowed <= hi:
                    for _t in range(3):
                        nv = rng.randint(min_allowed, hi)
                        if try_move(ri, pi, nv):
                            break
        return best_rows, best_val

    # ---------- branch-and-bound over candidate pool ----------
    def branch_and_bound(pool, n_rows, k, scope_limit, rng, start_time, deadline):
        if not pool:
            return None
        items = list(pool.items())
        # sort by max ascending to find small scope first
        items.sort(key=lambda kv: kv[0][-1])
        # prune to those within scope_limit
        items = [(t, d) for t, d in items if t[-1] <= scope_limit]
        if len(items) < n_rows:
            return None
        N = len(items)
        # conflict bitsets
        bitsets = []
        for idx in range(N):
            d = items[idx][1]
            bs = 0
            for jdx in range(N):
                if jdx != idx and (d & items[jdx][1]):
                    bs |= (1 << jdx)
            bitsets.append(bs)
        # greedy upper bound: max additional rows we can still pick from a set of candidates
        # We'll do a recursive branch-and-bound with ordering by index.
        best_found = [None]
        best_scope_bb = [scope_limit + 1]

        # order items by max so we can stop exploring once scope exceeds best
        # Precompute suffix feasibility: from index i, can we still reach n_rows?
        # Simple: just recurse.

        sys.setrecursionlimit(10000)

        chosen = []
        chosen_diffs = set()

        def bound(i, remaining):
            # upper bound on how many more we can pick from items[i:]
            # optimistic: all remaining
            return N - i

        def dfs(i, remaining, current_scope):
            if time.time() > deadline - 15:
                return
            if remaining == 0:
                # found a complete set
                sc = current_scope
                if sc < best_scope_bb[0]:
                    best_scope_bb[0] = sc
                    best_found[0] = [items[t][0] for t in chosen]
                return
            if i >= N:
                return
            # prune: not enough items left
            if N - i < remaining:
                return
            # prune: current_scope already >= best
            if current_scope >= best_scope_bb[0]:
                return
            # branch: include item i or skip
            # include if no conflict with chosen
            t = items[i][0]
            d = items[i][1]
            conflict = False
            for ct in chosen:
                if d & items[ct[1]][1]:  # ct stores (tuple, idx)
                    conflict = True
                    break
            # We stored chosen as list of (tuple, idx)
            if not conflict:
                new_scope = max(current_scope, t[-1])
                if new_scope < best_scope_bb[0]:
                    chosen.append((t, i))
                    dfs(i + 1, remaining - 1, new_scope)
                    chosen.pop()
            # skip
            dfs(i + 1, remaining, current_scope)

        # To make this tractable, limit pool size and use a better ordering + greedy seed
        # Greedy seed to set an initial best_scope_bb
        greedy_chosen = []
        greedy_diffs = set()
        for idx in range(N):
            if len(greedy_chosen) >= n_rows:
                break
            t = items[idx][0]
            d = items[idx][1]
            if d & greedy_diffs:
                continue
            greedy_chosen.append((t, idx))
            greedy_diffs |= d
        if len(greedy_chosen) == n_rows:
            best_scope_bb[0] = max(t[-1] for t, _ in greedy_chosen)
            best_found[0] = [t for t, _ in greedy_chosen]

        # Limit the search: only recurse over a subset to keep time bounded
        # Use a time-limited DFS with a node budget
        node_budget = [200000]
        def dfs2(i, remaining, current_scope):
            if time.time() > deadline - 20:
                return
            if node_budget[0] <= 0:
                return
            node_budget[0] -= 1
            if remaining == 0:
                sc = current_scope
                if sc < best_scope_bb[0]:
                    best_scope_bb[0] = sc
                    best_found[0] = [items[t][0] for t in chosen]
                return
            if i >= N:
                return
            if N - i < remaining:
                return
            if current_scope >= best_scope_bb[0]:
                return
            t = items[i][0]
            d = items[i][1]
            conflict = False
            for ct in chosen:
                if d & items[ct[1]][1]:
                    conflict = True
                    break
            if not conflict:
                new_scope = max(current_scope, t[-1])
                if new_scope < best_scope_bb[0]:
                    chosen.append((t, i))
                    dfs2(i + 1, remaining - 1, new_scope)
                    chosen.pop()
            dfs2(i + 1, remaining, current_scope)

        # Cap N to keep bitsets manageable
        if N > 300:
            # keep the 300 smallest-scope items
            items = items[:300]
            N = len(items)
            bitsets = []
            for idx in range(N):
                d = items[idx][1]
                bs = 0
                for jdx in range(N):
                    if jdx != idx and (d & items[jdx][1]):
                        bs |= (1 << jdx)
                bitsets.append(bs)

        chosen = []
        dfs2(0, n_rows, 0)
        if best_found[0] is not None:
            return [list(r) for r in best_found[0]]
        return None

    # ---------- main search ----------
    # 1) Build a candidate pool bounded by current best scope
    scope_limit = best_scope
    target_pool = min(250, 40 * n_rows)
    rng = random.Random(12345)
    pool = build_pool(n_rows, k, scope_limit, target_pool, rng, start_time, deadline)

    # 2) Try branch-and-bound
    if _time_left() > 30:
        bb_result = branch_and_bound(pool, n_rows, k, scope_limit, rng, start_time, deadline)
        if bb_result is not None:
            if _validate(bb_result):
                sc = _global_max(bb_result)
                if sc < best_scope:
                    best_scope = sc
                    best_solution = [r[:] for r in bb_result]

    # 3) Refine with local search (move-based)
    if _time_left() > 15:
        rows = [r[:] for r in best_solution]
        row_diffs = [_row_diffs(r) for r in rows]
        all_used = set()
        for dset in row_diffs:
            all_used |= dset
        row_maxes = [max(r) for r in rows]
        for trial in range(4):
            if _time_left() < 15:
                break
            iters = max(2000, int(_time_left() * 300))
            rs, sc = local_search_move(rows, all_used, row_diffs, row_maxes,
                                       n_rows, k, rng, start_time, deadline, iters)
            if sc < best_scope:
                best_scope = sc
                best_solution = rs
                rows = [r[:] for r in best_solution]
                row_diffs = [_row_diffs(r) for r in rows]
                all_used = set()
                for dset in row_diffs:
                    all_used |= dset
                row_maxes = [max(r) for r in rows]
            else:
                # re-sync state
                rows = [r[:] for r in best_solution]
                row_diffs = [_row_diffs(r) for r in rows]
                all_used = set()
                for dset in row_diffs:
                    all_used |= dset
                row_maxes = [max(r) for r in rows]

    # 4) Final polish with more local search
    if _time_left() > 5:
        rows = [r[:] for r in best_solution]
        row_diffs = [_row_diffs(r) for r in rows]
        all_used = set()
        for dset in row_diffs:
            all_used |= dset
        row_maxes = [max(r) for r in rows]
        iters = max(500, int(_time_left() * 400))
        rs, sc = local_search_move(rows, all_used, row_diffs, row_maxes,
                                   n_rows, k, rng, start_time, deadline, iters)
        if sc < best_scope:
            best_scope = sc
            best_solution = rs

    # ensure validity of final answer
    if not _validate(best_solution):
        best_solution = greedy_construct(n_rows, k)

    return best_solution
# EVOLVE-BLOCK-END
