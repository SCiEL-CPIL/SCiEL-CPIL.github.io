"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start = time.time()
    DEADLINE = start + 86.0  # safety margin under 90s wall-clock

    n = 128

    # primes <= n
    primes = []
    for x in range(2, n + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    prime_index = {p: i for i, p in enumerate(primes)}
    P = len(primes)

    # exp[pi][x] = exponent of primes[pi] in x
    exp = [[0] * (n + 1) for _ in range(P)]
    for x in range(2, n + 1):
        v = x
        for pi in range(P):
            p = primes[pi]
            if p * p > v:
                break
            while v % p == 0:
                exp[pi][x] += 1
                v //= p
        if v > 1:
            exp[prime_index[v]][x] += 1

    maxexp = [0] * P
    for pi in range(P):
        for x in range(1, n + 1):
            if exp[pi][x] > maxexp[pi]:
                maxexp[pi] = exp[pi][x]

    # gt[pi][t] = bitmask of c with exp[pi][c] > t
    # ne[pi][t] = bitmask of c with exp[pi][c] != t
    gt = {}
    ne = {}
    for pi in range(P):
        gt[pi] = [0] * (maxexp[pi] + 1)
        ne[pi] = [0] * (maxexp[pi] + 1)
        for t in range(maxexp[pi] + 1):
            m_gt = 0
            m_ne = 0
            for c in range(1, n + 1):
                e = exp[pi][c]
                if e > t:
                    m_gt |= (1 << c)
                if e != t:
                    m_ne |= (1 << c)
            gt[pi][t] = m_gt
            ne[pi][t] = m_ne

    # forbidden[a][b] = bitmask of c such that (a,b,c) is a bad triple
    forbidden = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            mask = 0
            for pi in range(P):
                ea = exp[pi][a]
                eb = exp[pi][b]
                if ea == eb:
                    mask |= gt[pi][ea]
                elif ea > eb:
                    mask |= ne[pi][ea]
                else:
                    mask |= ne[pi][eb]
            mask &= ~((1 << a) | (1 << b))
            forbidden[a][b] = mask
            forbidden[b][a] = mask

    def union_of(S):
        u = 0
        L = len(S)
        for i in range(L):
            ai = S[i]
            for j in range(i + 1, L):
                u |= forbidden[ai][S[j]]
        return u

    def valid(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return (union_of(S) & m) == 0

    # helpers for orderings
    def omega(x):
        c = 0
        v = x
        for pi in range(P):
            e = exp[pi][x]
            c += e
        return c

    def lpf(x):
        m = 1
        for pi in range(P):
            if exp[pi][x]:
                m = max(m, primes[pi])
        return m

    best = None
    best_size = 0

    def consider(S):
        nonlocal best, best_size
        if S and len(S) > best_size:
            best_size = len(S)
            best = list(S)

    # ---- Phase 1: many greedy restarts ----
    orderings = []
    orderings.append(list(range(1, n + 1)))
    orderings.append(list(range(n, 0, -1)))
    orderings.append(sorted(range(1, n + 1), key=omega))
    orderings.append(sorted(range(1, n + 1), key=omega, reverse=True))
    orderings.append(sorted(range(1, n + 1), key=lpf))
    orderings.append(sorted(range(1, n + 1), key=lpf, reverse=True))
    orderings.append(sorted(range(1, n + 1), key=lambda x: (omega(x), lpf(x))))
    orderings.append(sorted(range(1, n + 1), key=lambda x: (lpf(x), -omega(x))))
    for p in [2, 3, 5, 7, 11, 13]:
        orderings.append(sorted(range(1, n + 1), key=lambda x: x % p))

    random.seed(12345)
    for _ in range(2500):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        orderings.append(perm)

    def greedy(order):
        S = []
        Sset = set()
        u = 0
        for x in order:
            if x in Sset:
                continue
            if (u >> x) & 1:
                continue
            for y in S:
                u |= forbidden[x][y]
            S.append(x)
            Sset.add(x)
        return S

    top = []
    for order in orderings:
        if time.time() > DEADLINE - 25.0:
            break
        S = greedy(order)
        consider(S)
        top.append(S)
        if len(top) > 40:
            top.sort(key=len, reverse=True)
            top = top[:40]

    # ---- Phase 2: local search on the best seeds ----
    top.sort(key=len, reverse=True)

    def local_search(S, time_limit):
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # (a) plain add
            u = union_of(list(S))
            for x in range(1, n + 1):
                if x in S:
                    continue
                if (u >> x) & 1:
                    continue
                S.add(x)
                improved = True
                break
            if improved:
                continue
            # (b) 1-swap then add
            Slist = list(S)
            done = False
            for y in Slist:
                if time.time() > time_limit:
                    break
                for x in range(1, n + 1):
                    if x in S:
                        continue
                    T = S - {y} | {x}
                    if not valid(T):
                        continue
                    uT = union_of(list(T))
                    for z in range(1, n + 1):
                        if z in T:
                            continue
                        if (uT >> z) & 1:
                            continue
                        S = T | {z}
                        improved = True
                        done = True
                        break
                    if done:
                        break
                if done:
                    break
            if done:
                continue
            # (c) sampled 2-swap then add
            Slist = list(S)
            done = False
            cand_out = [x for x in range(1, n + 1) if x not in S]
            random.shuffle(Slist)
            random.shuffle(cand_out)
            for i in range(min(len(Slist), 12)):
                for j in range(i + 1, min(len(Slist), 12)):
                    if time.time() > time_limit:
                        break
                    y1, y2 = Slist[i], Slist[j]
                    for x1 in cand_out[:24]:
                        for x2 in cand_out[:24]:
                            if x1 == x2:
                                continue
                            T = S - {y1, y2} | {x1, x2}
                            if not valid(T):
                                continue
                            uT = union_of(list(T))
                            for z in range(1, n + 1):
                                if z in T:
                                    continue
                                if (uT >> z) & 1:
                                    continue
                                S = T | {z}
                                improved = True
                                done = True
                                break
                            if done:
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
        return S

    for S in top[:12]:
        if time.time() > DEADLINE - 4.0:
            break
        S2 = local_search(set(S), DEADLINE - 4.0)
        consider(S2)
        if best_size >= n:
            break

    # defensive validation + fallback
    if best is None:
        best = [1]
    elif not valid(set(best)):
        # rebuild a guaranteed-valid set via greedy
        best = greedy(list(range(1, n + 1)))
        if not best:
            best = [1]

    return best
# EVOLVE-BLOCK-END
