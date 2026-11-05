import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s limit

    NEED = set(range(1, 31))
    NEED_MASK = sum(1 << (d - 1) for d in NEED)  # bits 0..29 for d=1..30
    MAXN = 32

    def check(a):
        """Return (valid, has_all_need, max_elem) for a set/list of ints containing 0.
        valid: all positive differences distinct (Sidon) and len<=MAXN and 0 in a.
        has_all_need: every d in 1..30 is a difference."""
        if 0 not in a:
            return False, False, 0
        a = sorted(set(a))
        if len(a) > MAXN:
            return False, False, 0
        seen = set()
        mask = 0
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                d = a[j] - ai
                if d in seen:
                    return False, False, 0
                seen.add(d)
                if d <= 30:
                    mask |= 1 << (d - 1)
        has_all = (mask & NEED_MASK) == NEED_MASK
        return True, has_all, a[-1]

    def is_valid(a):
        v, h, _ = check(a)
        return v and h

    def fast_greedy(M, seed=0, skip_prob=0.0):
        """Greedy build within [0, M]: add smallest x maintaining Sidon, stop when all needs met.
        Returns list or None if not all needs met within M."""
        rng = random.Random(seed)
        A = [0]
        diffs = set()
        need_remaining = set(NEED)
        for x in range(1, M + 1):
            ok = True
            for a in A:
                d = x - a
                if d in diffs:
                    ok = False
                    break
            if not ok:
                continue
            if skip_prob > 0 and rng.random() < skip_prob:
                continue
            # tentatively add
            new_diffs = diffs.copy()
            for a in A:
                new_diffs.add(x - a)
            A.append(x)
            diffs = new_diffs
            # update need
            for a in A:
                d = x - a
                if d in need_remaining:
                    need_remaining.discard(d)
            if not need_remaining:
                return A
        return None

    def backtracking_search(M, time_limit, seed=0, max_nodes=2000000):
        """DFS to find a Sidon set within [0, M] containing 0, with all diffs 1..30.
        We choose up to MAXN elements. Returns list or None."""
        rng = random.Random(seed)
        A = [0]
        diffs = set()
        need_remaining = set(NEED)
        nodes = 0
        t0 = time.time()
        best = [None]

        def dfs(start, depth):
            nonlocal nodes
            if best[0] is not None:
                return True
            if time.time() - t0 > time_limit:
                return True
            if not need_remaining:
                best[0] = list(A)
                return True
            if depth >= MAXN:
                return False
            # prune: not enough room to add elements to cover remaining needs
            # each added element can cover at most (depth) new small diffs but we just bound by M
            # candidate range
            cand_lo = start
            cand_hi = M
            if cand_lo > cand_hi:
                return False
            # choose candidates: prefer small values but also those covering needs
            # Build list of candidate x in [cand_lo, cand_hi] that are conflict-free
            cands = []
            for x in range(cand_lo, cand_hi + 1):
                ok = True
                for a in A:
                    d = x - a
                    if d in diffs:
                        ok = False
                        break
                if ok:
                    cands.append(x)
            if not cands:
                return False
            # Order candidates by how many needs they cover (desc), then value (asc)
            def cover_count(x):
                c = 0
                for a in A:
                    d = x - a
                    if d in need_remaining:
                        c += 1
                return c
            cands.sort(key=lambda x: (-cover_count(x), x))
            # Limit branching: try top few plus some random
            trial = cands[:8]
            if len(cands) > 8:
                extra = rng.sample(cands[8:], min(4, len(cands) - 8))
                trial += extra
            for x in trial:
                nodes += 1
                # add x
                new_diffs = diffs.copy()
                for a in A:
                    new_diffs.add(x - a)
                A.append(x)
                old_need = set(need_remaining)
                for a in A:
                    d = x - a
                    if d in need_remaining:
                        need_remaining.discard(d)
                if dfs(x + 1, depth + 1):
                    return True
                # backtrack
                need_remaining = old_need
                A.pop()
            return False

        dfs(1, 1)
        return best[0]

    def local_search_reduce(a, time_limit, seed=0):
        """Given a valid set, try to reduce max by moving elements to smaller values."""
        rng = random.Random(seed)
        cur = sorted(a)
        cur_max = max(cur)
        best = list(cur)
        best_max = cur_max
        t0 = time.time()
        while time.time() - t0 < time_limit:
            improved = False
            # try to move largest element down
            for idx in range(len(cur) - 1, 0, -1):
                old_val = cur[idx]
                for new_val in range(old_val - 1, 0, -1):
                    if new_val in cur:
                        continue
                    test = list(cur)
                    test[idx] = new_val
                    test = sorted(set(test))
                    v, h, mx = check(test)
                    if v and h and mx < best_max:
                        cur = test
                        best = list(cur)
                        best_max = mx
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # random perturbation
                idx = rng.randint(1, len(cur) - 1)
                delta = rng.randint(-12, 12)
                new_val = cur[idx] + delta
                if new_val <= 0 or new_val in cur:
                    continue
                test = list(cur)
                test[idx] = new_val
                test = sorted(set(test))
                v, h, mx = check(test)
                if v and h and mx < best_max:
                    cur = test
                    best = list(cur)
                    best_max = mx
            if not improved:
                break
        return best

    # Phase 0: quick greedy to get an initial feasible solution (upper bound)
    best_a = None
    best_max = 10**6
    for sp in [0.0, 0.1, 0.2, 0.3]:
        a = fast_greedy(5000, seed=12345 + sp * 1000, skip_prob=sp)
        if a is not None and is_valid(a):
            m = max(a)
            if m < best_max:
                best_max = m
                best_a = a
    if best_a is None:
        # fallback: plain greedy with larger bound
        best_a = fast_greedy(20000, seed=999, skip_prob=0.0)
        if best_a is None:
            # last resort: brute small
            return list(range(0, 31))
        best_max = max(best_a)

    # Phase 1: binary search on M using backtracking_search
    lo = 1
    hi = best_max
    while lo < hi and time.time() < deadline - 15:
        mid = (lo + hi) // 2
        found = backtracking_search(mid, time_limit=min(4.0, deadline - time.time() - 15), seed=mid * 7 + 1)
        if found is not None and is_valid(found):
            hi = mid
            if max(found) < best_max:
                best_max = max(found)
                best_a = found
        else:
            lo = mid + 1

    # Phase 2: local search to shave max
    if best_a is not None and time.time() < deadline - 5:
        best_a = local_search_reduce(best_a, time_limit=min(8.0, deadline - time.time() - 5), seed=42)
        best_max = max(best_a)

    # Phase 3: a few restarts of backtracking at hi to possibly find smaller
    if time.time() < deadline - 3:
        for s in range(5):
            if time.time() > deadline - 3:
                break
            found = backtracking_search(best_max - 1, time_limit=min(3.0, deadline - time.time() - 3), seed=1000 + s * 137)
            if found is not None and is_valid(found) and max(found) < best_max:
                best_max = max(found)
                best_a = found

    if best_a is not None and is_valid(best_a):
        return best_a

    # Final fallback
    return list(range(0, 31))
