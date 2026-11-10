import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s/150s limit

    def now():
        return time.time()

    def diffs_of(row):
        s = set()
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                s.add(row[j] - ai)
        return s

    def scope_of(rows):
        return max(r[-1] for r in rows)

    # ---------- Fast randomized greedy (initial solution + restarts) ----------
    def greedy_construct(seed, scope_limit=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if scope_limit is not None and x > scope_limit:
                    return None
                new = {x - a for a in row}
                if new & used:
                    continue
                # light randomization: occasionally skip a valid small candidate
                if rng.random() < 0.15:
                    continue
                row.append(x)
                used |= new
            rows.append(row)
        return rows

    # ---------- Initial upper bound ----------
    best_rows = None
    best_scope = float('inf')
    for seed in range(40):
        if now() > deadline:
            break
        rows = greedy_construct(seed)
        if rows is not None:
            sc = scope_of(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # deterministic pure greedy fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_scope = scope_of(best_rows)

    # ---------- ILP-style branch-and-bound over candidate rows ----------
    # Binary variable per candidate row; constraint: no difference used twice.
    # Pruning via difference-count capacity (sparsity-aware).

    def candidate_rows_up_to(limit):
        """All valid rows (size k+1, start 0, max<=limit) with disjoint diffs internally."""
        res = []
        # DFS building a single row; collect all valid completions up to limit.
        # To bound memory, cap number of candidates.
        max_candidates = 200000

        def dfs(row, used):
            if now() > deadline:
                return
            if len(res) >= max_candidates:
                return
            if len(row) == k + 1:
                res.append((row[:], frozenset(used)))
                return
            last = row[-1]
            rem = (k + 1) - len(row)
            # smallest possible last element if we add rem more: last+rem
            # so current x must allow room: x + (rem-1) <= limit
            max_x = limit - (rem - 1)
            if last + 1 > max_x:
                return
            for x in range(last + 1, max_x + 1):
                new = {x - a for a in row}
                if new & used:
                    continue
                row.append(x)
                used |= new
                dfs(row, used)
                used.difference_update(new)
                row.pop()

        dfs([0], set())
        return res

    def branch_and_bound(scope_limit, seed, time_budget):
        """Select n_rows candidate rows with disjoint diffs, all max<=scope_limit.
        Returns rows or None."""
        rng = random.Random(seed)
        cands = candidate_rows_up_to(scope_limit)
        if len(cands) < n_rows:
            return None

        # Precompute diff sets
        cdiffs = [cd for _, cd in cands]
        crow = [r for r, _ in cands]
        ndiff_per_row = (k + 1) * k // 2

        # Heuristic lower bound: greedy cheapest (fewest new diffs) selection
        def greedy_lower_bound(used):
            remaining = n_rows
            # quick greedy
            cnt = 0
            used_local = set(used)
            for cd in cdiffs:
                if remaining == 0:
                    break
                if not (cd & used_local):
                    used_local |= cd
                    cnt += 1
                    remaining -= 1
            return cnt

        # We want to find a feasible selection; use DFS with capacity pruning.
        used = set()
        chosen = []
        # Order candidates by "scarcity": rows whose diffs are rare are more valuable.
        # Compute frequency of each diff across candidates.
        freq = {}
        for cd in cdiffs:
            for d in cd:
                freq[d] = freq.get(d, 0) + 1
        # Sort candidates: prefer those with rarer diffs (higher min rarity)
        rarity = [min(freq[d] for d in cd) for cd in cdiffs]
        order = sorted(range(len(cdiffs)), key=lambda i: (-rarity[i], i))
        cdiffs = [cdiffs[i] for i in order]
        crow = [crow[i] for i in order]
        n_cands = len(cdiffs)

        def dfs(idx, depth):
            if now() > deadline:
                return None
            if depth == n_rows:
                return chosen[:]
            # Capacity pruning: remaining capacity in diffs
            # Each additional row needs ndiff_per_row new diffs, all <= scope_limit
            # Total possible distinct diffs within [1, scope_limit] is scope_limit
            # If used count + ndiff_per_row*(n_rows-depth) > scope_limit -> infeasible
            remaining_rows = n_rows - depth
            if len(used) + ndiff_per_row * remaining_rows > scope_limit:
                return None
            # Try candidates from idx
            for j in range(idx, n_cands):
                if now() > deadline:
                    return None
                cd = cdiffs[j]
                if cd & used:
                    continue
                chosen.append(crow[j])
                used |= cd
                res = dfs(j + 1, depth + 1)
                if res is not None:
                    return res
                used.difference_update(cd)
                chosen.pop()
            return None

        return dfs(0, 0)

    # Try to beat the greedy scope via branch-and-bound at decreasing scope limits.
    # Only attempt if we have a reasonable best_scope to start from.
    if best_scope <= 2000:  # avoid huge candidate generation
        # Try scope = best_scope - 1, -2, ... down to a floor, with time budget each
        trial_scope = best_scope - 1
        floor = max(105, best_scope - 25)  # counting bound is 105; don't go far below
        t = 0
        while trial_scope >= floor and now() < deadline:
            t += 1
            budget = min(8.0, (deadline - now()) * 0.5)
            if now() + budget > deadline:
                break
            seed = 1000 + t
            res = branch_and_bound(trial_scope, seed, budget)
            if res is not None:
                sc = scope_of(res)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [r[:] for r in res]
                    trial_scope = best_scope - 1  # continue trying to improve
                else:
                    trial_scope -= 1
            else:
                # failed at this scope; try a smaller scope
                trial_scope -= 1

    # Final safety: ensure we always return a valid best_rows
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)

    return best_rows
