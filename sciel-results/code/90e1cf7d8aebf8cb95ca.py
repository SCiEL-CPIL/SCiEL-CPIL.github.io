import time
import random

def construct():
    deadline = time.time() + 84.0  # safety margin before 90s wall-clock
    N = 128
    base = set(range(65, N + 1))   # 64 elements, internally safe (all pair-sums > 128)

    def is_safe_to_add(S, x):
        # x not in S. Conflict iff distinct a,b in S with x+a,x+b,a+b all in S.
        A = set()
        for a in S:
            if (x + a) in S:
                A.add(a)
        # need distinct a,b in A with a+b in S
        for a in A:
            Sa = {s - a for s in S}   # b such that a+b in S
            inter = A & Sa
            inter.discard(a)
            if inter:
                return False
        return True

    def full_valid(S):
        el = sorted(S)
        m = len(el)
        for i in range(m):
            a = el[i]
            for j in range(i + 1, m):
                b = el[j]
                ab = a + b
                for k in range(j + 1, m):
                    c = el[k]
                    if ab in S and (a + c) in S and (b + c) in S:
                        return False
        return True

    pool = list(range(1, 65))

    def greedy(order):
        S = set(base)
        for x in order:
            if time.time() > deadline:
                break
            if x not in S and is_safe_to_add(S, x):
                S.add(x)
        return S

    def add_any(S):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in pool:
                if x not in S and is_safe_to_add(S, x):
                    S.add(x)
                    changed = True
        return S

    best = set(base)
    best_size = len(best)

    # multiple greedy orders
    orders = [pool, list(reversed(pool))]
    rng = random.Random(20240517)
    restarts = 0
    while time.time() < deadline - 12.0 and restarts < 250:
        o = pool[:]
        rng.shuffle(o)
        orders.append(o)
        restarts += 1

    for o in orders:
        if time.time() > deadline - 10.0:
            break
        S = greedy(o)
        S = add_any(S)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # local search: 1-remove + refill
    def t_elems(S):
        return [x for x in S if x <= 64]

    improved = True
    while improved and time.time() < deadline - 6.0:
        improved = False
        for e in list(t_elems(best)):
            if time.time() > deadline - 6.0:
                break
            S2 = set(best)
            S2.discard(e)
            S2 = add_any(S2)
            if len(S2) > best_size:
                best, best_size = S2, len(S2)
                improved = True

    # local search: 2-remove + refill (limited)
    if time.time() < deadline - 5.0:
        te = t_elems(best)
        for i in range(len(te)):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, len(te)):
                if time.time() > deadline - 5.0:
                    break
                S2 = set(best)
                S2.discard(te[i])
                S2.discard(te[j])
                S2 = add_any(S2)
                if len(S2) > best_size:
                    best, best_size = S2, len(S2)

    # final safety verification
    if not full_valid(best):
        best = set(base)
        best_size = len(best)

    return [int(x) for x in sorted(best)]
