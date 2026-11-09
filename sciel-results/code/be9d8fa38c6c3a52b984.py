import time
import random

def construct():
    deadline = time.time() + 87.0
    N = 512
    MAXC = 48
    cache = {}

    def build_dp(S):
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for a in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def get_dps(S):
        key = tuple(sorted(S))
        r = cache.get(key)
        if r is not None:
            return r
        dp_full = build_dp(S)
        maxfull = sum(S)
        dp_without = {}
        n = len(S)
        for i in range(n):
            sub = S[:i] + S[i + 1:]
            dp_without[S[i]] = (build_dp(sub), maxfull - S[i])
        r = (dp_full, maxfull, dp_without)
        cache[key] = r
        return r

    def test_add(S, x, dp_full, maxfull, dp_without):
        n = len(S)
        # Check 1: x not the mean of any subset of S (size>=2)
        for k in range(2, n + 1):
            t = k * x
            if t > maxfull:
                break
            if (dp_full[k] >> t) & 1:
                return False
        # Check 2: no a in S becomes mean of a subset containing x
        for a in S:
            dpo, maxo = dp_without[a]
            m = n - 1
            for j in range(1, m + 1):
                t = (j + 1) * a - x
                if t < 0 or t > maxo:
                    continue
                if (dpo[j] >> t) & 1:
                    return False
        return True

    def verify(S):
        S = list(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i + 1:]
            m = len(sub)
            if m < 2:
                continue
            dp = build_dp(sub)
            maxs = sum(sub)
            for k in range(2, m + 1):
                t = k * a
                if t > maxs:
                    break
                if (dp[k] >> t) & 1:
                    return False
        return True

    # --- Conflict graph formulation ---
    # Edge between a,b (a<b) if they form a "dangerous pair":
    # a is the mean of {b, c} for some c, i.e., 2a = b + c => c = 2a - b
    # or b is the mean of {a, c} => c = 2b - a
    # More generally, in a set S, element x is invalid if x = mean of subset T of others.
    # For pairwise: if we pick both a and b, then for any other picked c:
    #   a is mean of {b,c} iff b+c = 2a iff c = 2a-b
    #   b is mean of {a,c} iff a+c = 2b iff c = 2b-a
    #   c is mean of {a,b} iff a+b = 2c iff c = (a+b)/2 (must be integer)
    # So a 3-element set {a,b,c} is "conflicting" if one is the mean of the other two.
    # For the full problem (subsets of size >= 2), we need the DP check, but
    # the pairwise 3-AP constraint is necessary. We use it for ordering.

    # Precompute: for each pair (a,b) with a<b, check if they're in a 3-AP
    # (i.e., there exists c such that {a,b,c} has one as mean of other two)
    # Actually, the direct conflict: if we have a and b, and later add c=2a-b or c=2b-a or c=(a+b)/2,
    # that creates a violation. But the conflict is really 3-way.
    # For greedy ordering, we use a simpler heuristic:
    # degree(a) = number of b != a such that a and b are "close" in AP sense
    # Specifically, a and b conflict if (a+b) is even and (a+b)//2 is in [1,N]
    # or if 2a-b is in [1,N] and != b, or 2b-a is in [1,N] and != a

    # Compute pairwise conflict degree for ordering
    # Two numbers a,b are "conflict-adjacent" if:
    #   (a+b) % 2 == 0 and 1 <= (a+b)//2 <= N and (a+b)//2 != a and (a+b)//2 != b
    #   or 2*a - b in [1,N] and 2*a - b != b and 2*a - b != a
    #   or 2*b - a in [1,N] and 2*b - a != a and 2*b - a != b
    # This means: if all three of {a, b, c} are selected where c is the "third" element,
    # then one is the mean of the other two.

    # For the greedy MIS heuristic, we want to pick elements with fewest conflicts first.
    # Define degree(a) = count of b in [1,N], b!=a, such that a and b are conflict-adjacent.

    conflict_adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # Check if a,b form part of a 3-AP
            s = a + b
            if s % 2 == 0:
                c = s // 2
                if 1 <= c <= N and c != a and c != b:
                    conflict_adj[a].add(b)
                    conflict_adj[b].add(a)
            d1 = 2 * a - b
            if 1 <= d1 <= N and d1 != a and d1 != b:
                conflict_adj[a].add(b)
                conflict_adj[b].add(a)
            d2 = 2 * b - a
            if 1 <= d2 <= N and d2 != a and d2 != b:
                conflict_adj[a].add(b)
                conflict_adj[b].add(a)

    degree = [len(conflict_adj[i]) for i in range(N + 1)]

    # Greedy MIS: repeatedly pick the vertex with minimum degree among remaining
    def greedy_mis():
        remaining = set(range(1, N + 1))
        selected = []
        # Work with a copy of degrees that we update
        deg = degree[:]
        while remaining and len(selected) < MAXC:
            # Pick element with minimum degree
            min_deg = min(deg[x] for x in remaining)
            candidates = [x for x in remaining if deg[x] == min_deg]
            # Tie-break: prefer larger numbers (they have fewer AP partners typically)
            x = max(candidates)
            selected.append(x)
            remaining.discard(x)
            # Remove neighbors from consideration (they conflict)
            for nb in conflict_adj[x]:
                if nb in remaining:
                    remaining.discard(nb)
                    # Update degrees
                    for nb2 in conflict_adj[nb]:
                        if nb2 in remaining:
                            deg[nb2] -= 1
            # Update degrees for remaining
            for nb in conflict_adj[x]:
                if nb in remaining:
                    deg[nb] -= 1
        return selected

    # Also try: greedy by adding in order of increasing degree, checking full validity
    def greedy_add_by_degree():
        order = sorted(range(1, N + 1), key=lambda x: (degree[x], -x))
        cur = []
        dps = None
        for x in order:
            if len(cur) >= MAXC:
                break
            if dps is None:
                dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)
                dps = None
        return cur

    # Try descending order too (large numbers have fewer AP partners)
    def greedy_add_desc():
        cur = []
        dps = None
        for x in range(N, 0, -1):
            if len(cur) >= MAXC:
                break
            if dps is None:
                dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)
                dps = None
        return cur

    # Try ascending
    def greedy_add_asc():
        cur = []
        dps = None
        for x in range(1, N + 1):
            if len(cur) >= MAXC:
                break
            if dps is None:
                dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)
                dps = None
        return cur

    # Random restarts with conflict-aware perturbation
    random.seed(20240611)

    def greedy_add_with_perturbation(order):
        cur = []
        dps = None
        for x in order:
            if len(cur) >= MAXC:
                break
            if dps is None:
                dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)
                dps = None
        return cur

    best = []
    best_sz = 0

    # Phase 1: Try multiple strategies
    strategies = []

    # Strategy A: Greedy MIS (conflict graph)
    if time.time() < deadline - 15:
        mis_set = greedy_mis()
        if verify(mis_set) and len(mis_set) > best_sz:
            best_sz = len(mis_set)
            best = mis_set[:]
        strategies.append(mis_set)

    # Strategy B: Greedy add by degree (ascending degree)
    if time.time() < deadline - 15:
        deg_set = greedy_add_by_degree()
        if verify(deg_set) and len(deg_set) > best_sz:
            best_sz = len(deg_set)
            best = deg_set[:]
        strategies.append(deg_set)

    # Strategy C: Greedy add descending
    if time.time() < deadline - 15:
        desc_set = greedy_add_desc()
        if verify(desc_set) and len(desc_set) > best_sz:
            best_sz = len(desc_set)
            best = desc_set[:]
        strategies.append(desc_set)

    # Strategy D: Greedy add ascending
    if time.time() < deadline - 15:
        asc_set = greedy_add_asc()
        if verify(asc_set) and len(asc_set) > best_sz:
            best_sz = len(asc_set)
            best = asc_set[:]
        strategies.append(asc_set)

    # Strategy E: Random restarts with conflict-aware ordering
    for trial in range(8):
        if time.time() > deadline - 10:
            break
        # Perturb the degree ordering
        order = list(range(1, N + 1))
        # Weight by inverse degree, add randomness
        weights = [1.0 / (degree[x] + 1) + random.random() * 0.5 for x in order]
        order = sorted(order, key=lambda x: weights[x - 1])
        rset = greedy_add_with_perturbation(order)
        if verify(rset) and len(rset) > best_sz:
            best_sz = len(rset)
            best = rset[:]
        strategies.append(rset)

    # Phase 2: Local search from best
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            dps = get_dps(S)
            # Try to add
            for u in sorted(unused, key=lambda x: degree[x]):
                if len(S) >= MAXC:
                    break
                if test_add(S, u, dps[0], dps[1], dps[2]):
                    S.append(u)
                    Sset.add(u)
                    unused.remove(u)
                    improved = True
                    break
            if len(S) >= MAXC:
                break
            # Swap attempts: remove one element, try to add two
            random.shuffle(unused)
            for u in unused[:50]:
                if time.time() > deadline - 3:
                    break
                for i, s in enumerate(S):
                    newS = S[:i] + S[i + 1:]
                    dps2 = get_dps(newS)
                    if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
                        S = newS + [u]
                        Sset.discard(s)
                        Sset.add(u)
                        if u in unused:
                            unused.remove(u)
                        unused.append(s)
                        improved = True
                        break
                if improved:
                    break
        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # Phase 3: Try combining - take union of top strategies and extract valid subset
    if best_sz < MAXC and time.time() < deadline - 3:
        # Merge all strategy results, try to build a larger set
        all_candidates = set()
        for st in strategies:
            all_candidates.update(st)
        # Sort by degree (lower degree first)
        merged_order = sorted(all_candidates, key=lambda x: degree[x])
        cur = []
        dps = None
        for x in merged_order:
            if len(cur) >= MAXC:
                break
            if dps is None:
                dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)
                dps = None
        if verify(cur) and len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]

    # Final safety: trim to a verified valid set
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
