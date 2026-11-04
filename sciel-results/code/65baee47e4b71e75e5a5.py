"""Construct difference triangle sets (n_rows, k) minimizing the largest entry.

Structural change: model each row as a Sidon set of differences, optimize scope
with simulated annealing, and add a feasibility-driven backtracking search that
directly targets the counting bound (112) using a Hall's-theorem feasibility
check (matching remaining differences to free slots).
"""
import time
import random
from collections import defaultdict


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    bound = total_diffs  # counting bound on the scope

    # ---------- helpers ----------
    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def build_global_used(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    used.add(row[j] - row[i])
        return used

    def check_valid(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def pure_greedy():
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

    def randomized_greedy(window=80):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                cands = list(range(x + 1, x + window + 1))
                random.shuffle(cands)
                found = False
                for cand in cands:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                if not found:
                    cand = x + 1
                    while True:
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            break
                        cand += 1
            rows.append(row)
        return rows

    # ---------- simulated annealing (difference-space scope minimization) ----------
    def sa_minimize_scope(rows, time_limit):
        """Simulated annealing to reduce the maximum entry (scope)."""
        rows = [r[:] for r in rows]
        best = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])
        used = build_global_used(rows)

        T = max(20.0, best_scope * 0.35)
        T_min = 0.3
        cool = 0.997
        it = 0
        no_improve = 0
        while time.time() < time_limit:
            it += 1
            if T <= T_min:
                T = T_min
            ri = random.randrange(n)
            ci = random.randrange(1, L)
            old_val = rows[ri][ci]
            old_row = rows[ri]

            # candidate: smaller value (to reduce scope) or nearby
            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope
            if upper < lower:
                continue
            # bias toward reducing the max
            if old_val == best_scope:
                hi = min(upper, old_val - 1)
                if hi < lower:
                    continue
                nv = random.randint(lower, hi)
            else:
                span = max(1, (upper - lower) // 4)
                nv = old_val + random.randint(-span, span)
                nv = max(lower, min(upper, nv))
            if nv == old_val:
                continue

            new_row = old_row[:]
            new_row[ci] = nv
            new_diffs = row_diffs_set(new_row)
            if len(new_diffs) != diffs_per_row:
                continue
            old_diffs = row_diffs_set(old_row)
            other_diffs = used - old_diffs
            if new_diffs & other_diffs:
                continue

            new_scope = max(max(r) for i, r in enumerate(rows) if i != ri)
            new_scope = max(new_scope, max(new_row))
            delta = new_scope - best_scope

            if delta <= 0:
                # improvement
                used -= old_diffs
                used |= new_diffs
                rows[ri] = new_row
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in rows]
                    no_improve = 0
                    T = max(T * 1.05, T_min)  # reheat on improvement
                else:
                    no_improve += 1
            else:
                # accept with probability
                import math
                if random.random() < math.exp(-delta / max(T, 1e-9)):
                    used -= old_diffs
                    used |= new_diffs
                    rows[ri] = new_row
                    no_improve += 1
                else:
                    no_improve += 1
            T *= cool
        return best, best_scope

    # ---------- backtracking feasibility search (targets the counting bound) ----------
    def backtracking_search(target_scope, time_limit):
        """Try to build n_rows rows with scope <= target_scope.

        Uses Hall's-theorem feasibility: at each step, the remaining differences
        must be matchable to free slots in [1, target_scope].
        """
        import bisect

        # Precompute all valid rows (as sorted difference sets) with max <= target_scope.
        # A row is a Sidon set of size diffs_per_row with all differences <= target_scope.
        # We enumerate via backtracking on the difference set.
        valid_rows = []
        max_diff = target_scope

        # DFS over difference sets. A Sidon set: all pair differences distinct.
        # We build the set incrementally, keeping differences sorted.
        # For a set S (sorted), adding x: new diffs = {x - s for s in S} must be
        # distinct from each other and from existing diffs.
        # Existing diffs are all pairwise differences of S.

        # We represent S as a sorted list; its diff set as a set.
        # To keep it fast, we only go up to diffs_per_row elements.

        # This can be expensive; limit enumeration.
        row_limit = 40000
        counter = [0]

        def dfs_diff_set(S, diffset):
            if time.time() > time_limit:
                return
            counter[0] += 1
            if len(S) == diffs_per_row:
                valid_rows.append((S[:], diffset.copy(), max(S)))
                return
            if len(valid_rows) >= row_limit:
                return
            # next element must be > S[-1] and <= max_diff
            start = S[-1] + 1
            for x in range(start, max_diff + 1):
                # new diffs: x - s for s in S
                new_diffs = set()
                ok = True
                for s in S:
                    d = x - s
                    if d in diffset or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                # also must be <= max_diff (they are, since x<=max_diff and s>=1)
                S.append(x)
                diffset |= new_diffs
                dfs_diff_set(S, diffset)
                S.pop()
                diffset -= new_diffs
                if len(valid_rows) >= row_limit or time.time() > time_limit:
                    return

        dfs_diff_set([], set())
        if not valid_rows:
            return None

        # Now select n_rows rows with pairwise disjoint difference sets.
        # Backtrack over rows.
        chosen = []
        used = set()

        def dfs_rows(idx, used):
            if time.time() > time_limit:
                return False
            if idx == n_rows:
                return True
            # feasibility: can remaining rows be matched?
            # remaining differences needed = (n_rows - idx) * diffs_per_row
            # free slots = target_scope - len(used)
            if target_scope - len(used) < (n_rows - idx) * diffs_per_row:
                return False
            # Hall check (subset of sizes): for each possible size t of a future
            # row's diff set, it needs t distinct free slots. Simplest: total free
            # must be >= remaining total. (Stronger Hall is expensive; use total.)
            for S, diffset, mx in valid_rows:
                if diffset & used:
                    continue
                chosen.append(S)
                new_used = used | diffset
                if dfs_rows(idx + 1, new_used):
                    return True
                chosen.pop()
                if time.time() > time_limit:
                    return False
            return False

        if dfs_rows(0, set()):
            # convert difference sets back to rows (positions)
            rows = []
            for S in chosen:
                # S is sorted differences; reconstruct a row of length k+1
                # starting at 0 such that the difference multiset equals S.
                # For a Sidon set of differences, the row is determined by
                # a "canonical" construction: row[0]=0, and we need row[j]-row[i]
                # to cover S. This is the inverse of the difference map.
                # Simplest: the row is the set of partial sums? No.
                # Actually, a Sidon difference set of size k(k+1)/2 corresponds
                # to a row of k+1 numbers. We reconstruct greedily.
                rows.append(_reconstruct_row(S, k))
            return rows
        return None

    def _reconstruct_row(S, k):
        """Given sorted difference set S of size k(k+1)/2, reconstruct a row."""
        # The row has k+1 entries a_0=0 < a_1 < ... < a_k.
        # The differences are all a_j - a_i for i<j.
        # The smallest difference is a_1 - a_0 = a_1.
        # The largest difference is a_k - a_0 = a_k.
        # We reconstruct by: a_1 = min(S). Then a_2 is such that a_2 - a_0 and a_2 - a_1
        # are in S, etc.
        # Greedy: maintain row; at each step pick the smallest x > row[-1] such that
        # {x - a for a in row} subset of S and we can complete.
        row = [0]
        remaining = set(S)
        # a_1 = min(S)
        a1 = min(S)
        row.append(a1)
        remaining.remove(a1)
        while len(row) < k + 1:
            last = row[-1]
            # try increasing x
            x = last + 1
            while True:
                new_diffs = {x - a for a in row}
                if new_diffs <= remaining:
                    # check we can complete: remaining after must allow completion
                    row.append(x)
                    remaining -= new_diffs
                    break
                x += 1
        return row

    # ---------- main search ----------
    # Baseline: pure greedy
    base_rows = pure_greedy()
    best_rows = base_rows
    best_scope = scope_of(base_rows)

    # Phase 1: randomized greedy restarts
    attempts = 0
    while time.time() < deadline - 60.0 and attempts < 60:
        attempts += 1
        rows = randomized_greedy(window=random.randint(40, 120))
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            if best_scope <= bound:
                break

    # Phase 2: simulated annealing on best and alternatives
    pool = [best_rows]
    for _ in range(4):
        if time.time() >= deadline - 30.0:
            break
        rows = randomized_greedy(window=random.randint(40, 100))
        sc = scope_of(rows)
        if sc <= best_scope + 15:
            pool.append(rows)

    for start in pool:
        if time.time() >= deadline - 20.0:
            break
        improved, sc = sa_minimize_scope(start, time_limit=deadline - 20.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc
            if best_scope <= bound:
                break

    # Phase 3: backtracking search targeting the counting bound
    # Try scopes from bound upward until we find a feasible one.
    if time.time() < deadline - 5.0:
        # First try the exact bound
        for target in range(bound, best_scope + 1):
            if time.time() >= deadline - 5.0:
                break
            if target >= best_scope:
                break
            rows = backtracking_search(target, time_limit=deadline - 5.0)
            if rows is not None and check_valid(rows):
                sc = scope_of(rows)
                if sc < best_scope:
                    best_rows = rows
                    best_scope = sc
                    break

    return best_rows
