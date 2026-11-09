import time
import random

def construct():
    deadline = time.time() + 82.0  # Safety margin from 90s limit
    N = 512
    MAXC = 48

    # ---- Precompute a conflict "degree" for each integer in [1, N]. ----
    # A number x conflicts with the set if it can be the mean of a subset of
    # at least two OTHER selected elements. The dominant, cheap proxy for how
    # "conflict-prone" a number is, is how many 3-term arithmetic progressions
    # (a, x, c) with a != x != c it sits at the MIDDLE of, i.e. pairs (a, c)
    # with a + c == 2*x. More such pairs => more likely x is an average of two
    # others => less "safe" to include. We also count pairs where x is an
    # endpoint (a + x = 2*c) as a lighter signal.
    def compute_conflict_scores(N):
        # mid[x] = number of distinct pairs (a,c), a<c, a+c==2x, a,c in [1,N], a!=x
        # end[x] = number of distinct pairs (a,c), a+c==2x where x is an endpoint
        #          i.e. x + c == 2*a or a + x == 2*c
        mid = [0] * (N + 1)
        end = [0] * (N + 1)
        for a in range(1, N + 1):
            for c in range(a + 1, N + 1):
                s = a + c
                if s % 2 != 0:
                    continue
                x = s // 2
                if 1 <= x <= N and x != a and x != c:
                    mid[x] += 1
                    # a and c are endpoints of a 3-AP centered at x
                    end[a] += 1
                    end[c] += 1
        # Combined score: middle role is the direct "mean of two" conflict.
        score = [0] * (N + 1)
        for x in range(1, N + 1):
            score[x] = 3 * mid[x] + end[x]
        return score

    conflict_score = compute_conflict_scores(N)

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
            max_j = min(m, (maxo + x) // a)
            for j in range(1, max_j + 1):
                t = (j + 1) * a - x
                if t < 0:
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

    base = list(range(1, N + 1))
    random.seed(20240611)

    # ---- Structural idea: conflict-aware ordering. ----
    # Order candidates by ascending conflict score (safest first). Ties broken
    # by a stable secondary key so the order is deterministic and reproducible.
    # This is the main structural change vs. the random-shuffle family.
    def degree_ordered():
        return sorted(base, key=lambda v: (conflict_score[v], v))

    def degree_ordered_desc():
        return sorted(base, key=lambda v: (conflict_score[v], -v))

    orders = []
    orders.append(degree_ordered())
    orders.append(degree_ordered_desc())
    orders.append(base)
    orders.append(base[::-1])
    # A few random perturbations of the degree order for diversity, but the
    # degree order remains the backbone.
    for _ in range(8):
        o = degree_ordered()
        random.shuffle(o)
        orders.append(o)

    best = []
    best_sz = 0

    # ---- Phase 1: multi-order greedy, degree-order first. ----
    for order in orders:
        if time.time() > deadline - 8:
            break
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
        if verify(cur) and len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
        if best_sz >= MAXC:
            break

    # ---- Phase 2: local search (add + swap) with degree-aware swap choice. ----
    # The swap heuristic is the refinement machinery: when we must remove an
    # element to make room, prefer removing the one with the HIGHEST conflict
    # score (most "dangerous"), and prefer inserting the unused candidate with
    # the LOWEST conflict score. This is a structural improvement over the
    # random swap order and directly leverages the precomputed conflict graph.
    if best_sz < MAXC and time.time() < deadline - 4:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            dps = get_dps(S)
            # Try adding unused elements (lowest conflict first)
            for u in sorted(unused, key=lambda v: conflict_score[v]):
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
            # Swap: insert a low-conflict unused u, remove a high-conflict s
            candidates = sorted(unused, key=lambda v: conflict_score[v])[:200]
            # Sort S by descending conflict score so we try removing the most
            # dangerous element first.
            s_order = sorted(range(len(S)), key=lambda i: -conflict_score[S[i]])
            for u in candidates:
                if time.time() > deadline - 2:
                    break
                for i in s_order:
                    s = S[i]
                    newS = S[:i] + S[i + 1:]
                    dps2 = get_dps(newS)
                    if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
                        S = newS + [u]
                        Sset.remove(s)
                        Sset.add(u)
                        unused.remove(u)
                        unused.append(s)
                        improved = True
                        break
                if improved:
                    break
        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # ---- Phase 3: extra local-search restarts with different seeds. ----
    if best_sz < MAXC and time.time() < deadline - 3:
        for seed in [42, 123, 456, 777]:
            if time.time() > deadline - 2:
                break
            random.seed(seed)
            S = best[:]
            Sset = set(S)
            unused = [x for x in range(1, N + 1) if x not in Sset]
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                dps = get_dps(S)
                for u in sorted(unused, key=lambda v: conflict_score[v]):
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
                candidates = sorted(unused, key=lambda v: conflict_score[v])[:150]
                s_order = sorted(range(len(S)), key=lambda i: -conflict_score[S[i]])
                for u in candidates:
                    if time.time() > deadline - 2:
                        break
                    for i in s_order:
                        s = S[i]
                        newS = S[:i] + S[i + 1:]
                        dps2 = get_dps(newS)
                        if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
                            S = newS + [u]
                            Sset.remove(s)
                            Sset.add(u)
                            unused.remove(u)
                            unused.append(s)
                            improved = True
                            break
                    if improved:
                        break
            if verify(S) and len(S) > best_sz:
                best_sz = len(S)
                best = S[:]
                if best_sz >= MAXC:
                    break

    # Final safety: trim to a verified valid set (removal preserves validity).
    if best and not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
