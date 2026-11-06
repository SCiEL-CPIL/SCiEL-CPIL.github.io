import math
import time
import itertools

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    n = 12

    # SPF table up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors(x):
        if x <= 1:
            return frozenset()
        s = set()
        while x > 1:
            p = spf[x]
            s.add(p)
            while x % p == 0:
                x //= p
        return frozenset(s)

    pf = {s: prime_factors(s) for s in range(2, MAX_SUM + 1)}

    def count(arr):
        u = set()
        for i in range(n - 1):
            ai = arr[i]
            for j in range(i + 1, n):
                u |= pf[ai + arr[j]]
        return len(u)

    # Build k-smooth pools for various prime sets
    def smooth_pool(primes, limit=10000):
        pool = set([1])
        for p in primes:
            new = set()
            for x in list(pool):
                v = x
                while v <= limit:
                    new.add(v)
                    v *= p
            pool |= new
        return sorted(pool)

    pool_7 = smooth_pool([2, 3, 5, 7])       # 7-smooth numbers
    pool_5 = smooth_pool([2, 3, 5])           # 5-smooth numbers

    # Candidate pools to search in
    candidate_pools = [pool_7, pool_5]

    def is_smooth_sum(u, v, smooth_set):
        s = u + v
        return s in smooth_set

    # For a given prime set P (as a frozenset), build smooth-sum set
    def build_smooth_sum_set(primes):
        P = frozenset(primes)
        ss = set()
        for s in range(2, MAX_SUM + 1):
            if pf[s] <= P:
                ss.add(s)
        return ss

    # Greedy clique / local search
    def local_search(start, pool, max_iter=500):
        cur = list(start)
        existing = set(cur)
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < deadline - 3.0:
            improved = False
            it += 1
            base = count(cur)
            for idx in range(n):
                if time.time() > deadline - 3.0:
                    break
                old = cur[idx]
                for cand in pool:
                    if cand > 10000:
                        break
                    if cand == old or cand in existing:
                        continue
                    cur[idx] = cand
                    sc = count(cur)
                    cur[idx] = old
                    if sc < base:
                        cur[idx] = cand
                        base = sc
                        existing.discard(old)
                        existing.add(cand)
                        improved = True
                        break
                if improved:
                    break
        return cur, count(cur)

    # Generate many seeds
    seeds = []

    # Seed A: greedy 12-clique in graph where edge = sum in smooth set
    for primes in ([2, 3, 5, 7], [2, 3, 5], [2, 3, 5, 7, 11], [2, 3, 5, 7, 11, 13]):
        ss = build_smooth_sum_set(primes)
        pool = smooth_pool(primes)
        # Greedy clique
        clique = []
        for u in pool:
            if u > 10000:
                break
            ok = True
            for v in clique:
                if (u + v) not in ss:
                    ok = False
                    break
            if ok:
                clique.append(u)
            if len(clique) >= n:
                break
        if len(clique) >= n:
            seeds.append(clique[:n])

    # Seed B: multiples of LCM-ish
    for base in (6, 30, 210, 420):
        multiples = [base * i for i in range(1, 100) if base * i <= 10000]
        if len(multiples) >= n:
            seeds.append(multiples[:n])
            # also shifted
            for off in (1, 2, 3, 5):
                shifted = [base * i + off for i in range(1, 100) if base * i + off <= 10000]
                if len(shifted) >= n:
                    seeds.append(shifted[:n])

    # Seed C: small 7-smooth
    seeds.append(pool_7[:n])
    seeds.append(pool_5[:n])

    # Seed D: all even small
    seeds.append([2 * i for i in range(1, n + 1)])
    seeds.append([2 * i + 1 for i in range(1, n + 1)])
    seeds.append(list(range(1, n + 1)))
    seeds.append([1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30])

    # Deduplicate & validate seeds
    valid_seeds = []
    seen = set()
    for s in seeds:
        if len(set(s)) != n or any(x > 10000 for x in s) or any(x <= 0 for x in s):
            continue
        key = tuple(sorted(s))
        if key in seen:
            continue
        seen.add(key)
        valid_seeds.append(list(s))

    best = None
    bs = 10 ** 9
    for s in valid_seeds:
        c = count(s)
        if c < bs:
            bs = c
            best = list(s)

    # Local search from best seed, then from each seed
    cur = list(best)
    for pool in candidate_pools:
        c, sc = local_search(cur, pool)
        if sc < bs:
            bs = sc
            best = c
            cur = list(c)
        if bs <= 5:
            return list(best)

    # Try local search from each valid seed
    for s in valid_seeds:
        if time.time() > deadline - 5.0:
            break
        for pool in candidate_pools:
            if time.time() > deadline - 4.0:
                break
            c, sc = local_search(list(s), pool, max_iter=200)
            if sc < bs:
                bs = sc
                best = c
                if bs <= 5:
                    return list(best)

    return list(best)
