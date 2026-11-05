import time

def construct():
    """Find a Sidon set (all positive differences distinct) containing 0 with
    every difference 1..30 realized, minimizing the largest element.

    Strategy:
      1. Build the "forced core": start from {0}, and repeatedly add the
         smallest positive element that realizes a still-missing difference
         without breaking the Sidon property. This yields a valid set with
         a guaranteed diameter (the seed's greedy result).
      2. Improve via iterative-deepening DFS backtracking with strong
         pruning (difference-set feasibility + Sidon feasibility), searching
         for a set with a smaller maximum element.
      3. Return the best (smallest max) set found before the deadline.
    """
    deadline = time.time() + 110.0   # safety margin under the 120 s limit

    # ---------- helpers ----------
    def is_sidon_set(S, used):
        """S sorted; used is set of already-taken differences. True iff adding
        nothing new is fine and S itself has all pairwise differences distinct
        (and none collides with used)."""
        n = len(S)
        seen = set(used)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                d = S[j] - a
                if d in seen:
                    return False
                seen.add(d)
        return True

    def core_greedy():
        """Greedy: realize each missing d with the smallest new element."""
        a = [0]
        used = set()
        def diffs_of(lst):
            return set(y - x for i, x in enumerate(lst) for y in lst[i + 1:])
        for d in range(1, 31):
            # current differences
            cur = diffs_of(a)
            if d in cur:
                continue
            # try to add a single element x so that d becomes a difference
            best = None
            # candidate x: for each a_i, x = a_i + d or x = a_i - d
            cands = set()
            for v in a:
                for s in (d, -d):
                    x = v + s
                    if x >= 0 and x not in a:
                        cands.add(x)
            for x in sorted(cands):
                trial = sorted(a + [x])
                td = diffs_of(trial)
                if len(td) == len(list(y - z for i, z in enumerate(trial) for y in trial[i + 1:])):
                    # all distinct
                    if d in td:
                        best = x
                        break
            if best is not None:
                a = sorted(a + [best])
            else:
                # fallback: add two far elements
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    td = list(y2 - y1 for i, y1 in enumerate(trial) for y2 in trial[i + 1:])
                    if len(set(td)) == len(td):
                        a = trial
                        break
                    y += 1
        return a

    best = core_greedy()
    best_max = max(best)

    # ---------- backtracking improvement ----------
    # We search for a set with max < best_max.
    # State: current sorted list S (always contains 0), used difference set.
    # We add elements in increasing order.

    def solve(limit):
        """Return a Sidon set (with 0, all diffs 1..30, max<=limit) or None."""
        # Precompute: all differences that must appear = 1..30
        need = set(range(1, 31))
        # We'll grow S. S[0]=0.
        S = [0]
        used = set()
        # track which needed diffs are already covered
        covered = set()

        # feasibility: can the remaining needed diffs be formed by pairs
        # among S plus up to (limit - max(S)) new larger elements?
        # Simple admissible bound: a set of size k in [0,limit] can produce
        # at most k*(k-1)/2 differences, all in [1,limit].
        # We need at least len(need) distinct diffs, and we can add at most
        # (limit - max(S)) more elements (values max(S)+1 .. limit).
        def feasible(S, used, need, limit):
            cur_max = S[-1]
            # diffs already covered
            # remaining needed not yet in used
            rem = need - used
            if not rem:
                return True
            # how many more elements can we add?
            max_size = limit - cur_max  # number of integer slots above cur_max
            k = len(S)
            total_possible = k + max_size
            # max distinct differences a set of total_possible elements can have
            max_diffs = total_possible * (total_possible - 1) // 2
            if max_diffs < len(rem):
                return False
            # each remaining needed diff must be <= limit
            if max(rem) > limit:
                return False
            return True

        def dfs(S, used, covered):
            if time.time() > deadline:
                return None
            if not (need - used):
                # all covered
                return list(S)
            if not feasible(S, used, need - used, limit):
                return None
            cur_max = S[-1]
            # choose next element x > cur_max, x <= limit
            # candidate ordering: prefer x that covers most needed diffs
            lo = cur_max + 1
            hi = limit
            # To keep search fast, try a limited window of candidates.
            # Prefer x that creates many needed differences.
            cands = []
            # We can also consider that the next element need not be consecutive.
            # Generate candidates by trying to cover the smallest missing diff.
            # For speed, iterate x over a subset: those x such that x - s in need
            # for some s in S (covers a needed diff), plus a few generic ones.
            seen_c = set()
            # For each needed diff d not yet covered, and each s in S, x = s + d
            # (and possibly x = s - d but x>cur_max so only s+d relevant mostly)
            for d in sorted(need - used):
                for s in S:
                    x = s + d
                    if lo <= x <= hi and x not in seen_c:
                        seen_c.add(x)
                        cands.append(x)
            # also include lo (smallest) to allow growth
            if lo not in seen_c:
                seen_c.add(lo)
                cands.append(lo)
            # sort candidates by how many needed diffs they cover (desc), then value
            def cover_count(x):
                cnt = 0
                for s in S:
                    d = x - s
                    if d in (need - used):
                        cnt += 1
                return cnt
            cands.sort(key=lambda x: (-cover_count(x), x))
            # limit candidate count for speed
            cands = cands[:60]
            for x in cands:
                # check Sidon: new differences x - s for s in S must be unused
                ok = True
                new_diffs = []
                for s in S:
                    d = x - s
                    if d in used:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                S.append(x)
                used2 = used | set(new_diffs)
                res = dfs(S, used2, covered | set(new_diffs))
                if res is not None:
                    return res
                S.pop()
            return None

        return dfs(S, used, covered)

    # Try improving: search for max < best_max, decreasing.
    # Start from best_max-1 downward, but cap the number of deep searches.
    # Use iterative deepening on the limit.
    search_lo = 1
    # A trivial lower bound: need 30 distinct diffs; with 0 and k elements the
    # smallest possible max for k elements is at least k-1 (consecutive). We need
    # at least enough elements. Keep search_lo modest.
    search_lo = max(search_lo, 1)

    # We'll try limits from best_max-1 down to search_lo, but with a time budget
    # and a cap on how many full DFS attempts we run.
    attempts = 0
    max_attempts = 40
    for lim in range(best_max - 1, search_lo - 1, -1):
        if time.time() > deadline:
            break
        if attempts >= max_attempts:
            break
        attempts += 1
        res = solve(lim)
        if res is not None:
            if max(res) < best_max:
                best = res
                best_max = max(res)
            break  # found smallest for this lim; stop (lim is decreasing)

    return best
