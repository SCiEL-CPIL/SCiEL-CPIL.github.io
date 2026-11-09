import time
import random
import numpy as np

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

    # Conflict-aware ordering: prioritize elements with fewer 3-term AP conflicts
    def count_conflicts(x, S):
        """Count how many elements in S form a 3-term AP with x"""
        cnt = 0
        Sset = set(S)
        for a in S:
            # Check if x is middle: a, x, b where b = 2x - a
            b = 2 * x - a
            if b in Sset and b != a and b != x:
                cnt += 1
            # Check if a is middle: x, a, b where b = 2a - x
            b = 2 * a - x
            if b in Sset and b != a and b != x:
                cnt += 1
            # Check if b is middle: a, b, x where b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in Sset and b != a and b != x:
                    cnt += 1
        return cnt

    base = list(range(1, N + 1))
    random.seed(20240611)
    orders = [base, base[::-1]]
    for _ in range(12):
        o = base[:]
        random.shuffle(o)
        orders.append(o)

    best = []
    best_sz = 0

    # Phase 1: multi-order greedy with conflict-aware reordering
    for order in orders:
        if time.time() > deadline - 8:
            break
        cur = []
        dps = None
        remaining = list(order)
        while remaining and len(cur) < MAXC:
            if time.time() > deadline - 8:
                break
            # Score each candidate by conflict count with current set
            if len(cur) >= 2:
                scored = [(count_conflicts(x, cur), x) for x in remaining]
                scored.sort()
                x = scored[0][1]
            else:
                x = remaining[0]
            if dps is None:
                dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)
                dps = None
                remaining.remove(x)
            else:
                remaining.remove(x)
        if verify(cur) and len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
        if best_sz >= MAXC:
            break

    # Phase 2: local search (add + swap)
    if best_sz < MAXC and time.time() < deadline - 4:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            dps = get_dps(S)
            # Try adding unused elements, prioritizing low-conflict ones
            if len(S) >= 2:
                scored = [(count_conflicts(u, S), u) for u in unused]
                scored.sort()
                candidates = [u for _, u in scored]
            else:
                candidates = list(unused)
            for u in candidates:
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
            # swap attempts: try removing high-conflict elements and adding low-conflict ones
            if len(S) >= 2:
                # Find element with most conflicts
                conflict_scores = [(count_conflicts(s, S), s) for s in S]
                conflict_scores.sort(reverse=True)
                swap_candidates = [s for _, s in conflict_scores[:10]]
            else:
                swap_candidates = S[:]
            random.shuffle(unused)
            for u in unused[:80]:
                if time.time() > deadline - 2:
                    break
                for s in swap_candidates:
                    if s not in S:
                        continue
                    newS = [x for x in S if x != s]
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

    # Final safety: trim to a verified valid set (removal preserves validity)
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
