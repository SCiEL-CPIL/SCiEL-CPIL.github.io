import time
import random

def construct():
    deadline = time.time() + 82.0
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

    # Phase 1: Multi-order greedy with diverse seeds
    base = list(range(1, N + 1))
    random.seed(42)
    
    # Generate diverse orders: ascending, descending, and several random
    orders = [base, base[::-1]]
    for _ in range(8):
        o = base[:]
        random.shuffle(o)
        orders.append(o)
    
    # Also try orders biased towards spread-out numbers
    for _ in range(4):
        o = base[:]
        # Interleave low and high
        half = len(o) // 2
        inter = []
        for i in range(half):
            inter.append(o[i])
            inter.append(o[N - 1 - i])
        if N % 2:
            inter.append(o[half])
        random.shuffle(inter)
        orders.append(inter)

    best = []
    best_sz = 0

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

    # Phase 2: Local search with aggressive swapping
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        
        # Sort unused by a heuristic: prefer numbers that are "spread out"
        # We'll just shuffle randomly for now
        
        improved = True
        max_iters = 200
        iters = 0
        while improved and time.time() < deadline - 3 and iters < max_iters:
            iters += 1
            improved = False
            dps = get_dps(S)
            
            # Try to add
            for u in unused[:]:
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
            
            # Try swaps: remove one, add one
            random.shuffle(unused)
            found_swap = False
            for u in unused[:150]:
                if time.time() > deadline - 3:
                    break
                for i in range(len(S)):
                    s = S[i]
                    newS = S[:i] + S[i + 1:]
                    # Check if we can add u to newS
                    dps2 = get_dps(newS)
                    if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
                        S = newS + [u]
                        Sset.remove(s)
                        Sset.add(u)
                        unused.remove(u)
                        unused.append(s)
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                improved = True
        
        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # Phase 3: More extensive local search with different swap strategies
    if best_sz < MAXC and time.time() < deadline - 2:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        
        for _ in range(30):
            if time.time() > deadline - 2:
                break
            # Randomly remove a few and try to add better ones
            if len(S) > 5:
                # Remove 1-2 random elements
                to_remove = random.sample(S, min(2, len(S) - 5))
                for s in to_remove:
                    S.remove(s)
                    Sset.remove(s)
                    unused.append(s)
                unused.sort()
                
                # Try to add back
                dps = get_dps(S)
                for u in unused[:]:
                    if len(S) >= MAXC:
                        break
                    if test_add(S, u, dps[0], dps[1], dps[2]):
                        S.append(u)
                        Sset.add(u)
                        unused.remove(u)
                        dps = None
                if verify(S) and len(S) > best_sz:
                    best_sz = len(S)
                    best = S[:]

    # Final safety
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
