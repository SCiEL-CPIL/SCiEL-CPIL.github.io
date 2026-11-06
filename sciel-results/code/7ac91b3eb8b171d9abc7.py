import math
import time

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    n = 12

    # Build SPF table for fast factorization
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
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

    # Build pool: 7-smooth numbers up to 10000
    pool = set()
    p2 = 1
    while p2 <= 10000:
        p3 = p2
        while p3 <= 10000:
            p5 = p3
            while p5 <= 10000:
                p7 = p5
                while p7 <= 10000:
                    pool.add(p7)
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
    pool = sorted(pool)

    # Seeds
    seeds = [
        [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30],
        [30 * i for i in range(1, 13)],
        [30 * i + 15 for i in range(1, 13)],
        [2 * i for i in range(1, 13)],
        [2 * i + 1 for i in range(1, 13)],
        pool[:12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27],
    ]

    best = None
    bs = 10**9
    for s in seeds:
        if len(set(s)) != 12 or any(x > 10000 for x in s) or any(x <= 0 for x in s):
            continue
        c = count(s)
        if c < bs:
            bs = c
            best = list(s)

    cur = list(best)
    improved = True
    it = 0
    corder = pool  # already sorted

    while improved and it < 2000 and time.time() < deadline - 2.0:
        improved = False
        it += 1
        base = count(cur)
        existing = set(cur)
        for idx in range(n):
            if time.time() > deadline - 2.0:
                return list(best)
            old = cur[idx]
            for cand in corder:
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
                if base < bs:
                    bs = base
                    best = list(cur)
                if bs <= 5:
                    return list(best)
                break

    return list(best)
