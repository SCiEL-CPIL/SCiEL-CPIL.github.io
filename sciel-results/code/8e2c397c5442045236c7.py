"""Simulated annealing with an exact incremental violation-count oracle.

Reformulation: S is invalid iff it contains a triple {a,b,c} (distinct) with
a+b, a+c, b+c all in S.  We track the exact number V of such triples and do
simulated annealing on energy E = V*BIG - |S|, using a fast bit-parallel update
for V when a single element is added/removed.
"""
import time
import math
import random

N = 128
BIG = 1000


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    # ---------- precompute ----------
    # partners[x]: y != x with x+y <= N
    partners = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        px = partners[x]
        for y in range(1, N + 1):
            if y != x and x + y <= N:
                px.append(y)
    # summask[a]: bitmask of b>a with a+b <= N
    summask = [0] * (N + 1)
    for a in range(1, N + 1):
        m = 0
        for b in range(a + 1, N + 1):
            if a + b <= N:
                m |= (1 << b)
        summask[a] = m

    def viol_involving(S, x):
        """# triples {a,b,x} that would be valid given current mask S (x not in S)."""
        Px = 0
        for y in partners[x]:
            if (S >> y) & 1 and (S >> (x + y)) & 1:
                Px |= (1 << y)
        if not Px:
            return 0
        cnt = 0
        m = Px
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            m ^= lsb
            cnt += (Px & summask[a] & S).bit_count()
        return cnt

    def total_V(S):
        s = 0
        m = S
        while m:
            lsb = m & (-m)
            x = lsb.bit_length() - 1
            m ^= lsb
            s += viol_involving(S & ~(1 << x), x)
        return s // 3

    def greedy_add(S):
        """Add any element that keeps V at 0 (order: small first)."""
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if not ((S >> x) & 1) and viol_involving(S, x) == 0:
                    S |= (1 << x)
                    changed = True
        return S

    def repair(S):
        """Remove elements until V == 0."""
        V = total_V(S)
        while V > 0:
            bestx = None
            bestc = -1
            m = S
            while m:
                lsb = m & (-m)
                x = lsb.bit_length() - 1
                m ^= lsb
                c = viol_involving(S & ~(1 << x), x)
                if c > bestc:
                    bestc = c
                    bestx = x
            S &= ~(1 << bestx)
            V -= bestc
        return S

    # ---------- initial valid set: upper block ----------
    up = 0
    for i in range(65, N + 1):
        up |= (1 << i)
    S0 = greedy_add(up)
    best = S0
    best_sz = S0.bit_count()

    rng = random.Random(20240517)

    def sa(S, V, T0, rng):
        nonlocal best, best_sz
        E = V * BIG - S.bit_count()
        T = T0
        while True:
            if time.time() > deadline - 4.0:
                break
            x = rng.randint(1, N)
            if (S >> x) & 1:
                S2 = S & ~(1 << x)
                V2 = V - viol_involving(S2, x)
                s2 = S2.bit_count()
            else:
                S2 = S | (1 << x)
                V2 = V + viol_involving(S, x)
                s2 = S.bit_count() + 1
            E2 = V2 * BIG - s2
            dE = E2 - E
            if dE < 0:
                accept = True
            elif dE < 50.0 * T and rng.random() < math.exp(-dE / T):
                accept = True
            else:
                accept = False
            if accept:
                S, V, E = S2, V2, E2
                if V == 0 and s2 > best_sz:
                    best = S
                    best_sz = s2
            T *= 0.9995
            if T < 0.02:
                T = 0.02
        return S, V

    # ---------- SA restarts with cooling schedules ----------
    for r in range(60):
        if time.time() > deadline - 6.0:
            break
        T0 = 60.0 / (r + 1)
        S, V = sa(S, V, T0, rng)
        S = repair(S)
        S = greedy_add(S)
        V = total_V(S)  # should be 0
        if V == 0 and S.bit_count() > best_sz:
            best = S
            best_sz = S.bit_count()

    # ---------- final polish ----------
    best = greedy_add(best)
    if total_V(best) != 0:
        best = repair(best)
        best = greedy_add(best)

    # ---------- defensive exact verification ----------
    def is_valid(S):
        elems = [i for i in range(1, N + 1) if (S >> i) & 1]
        Sset = set(elems)
        k = len(elems)
        for i in range(k):
            a = elems[i]
            for j in range(i + 1, k):
                b = elems[j]
                for c in elems[j + 1:]:
                    if (a + b) in Sset and (a + c) in Sset and (b + c) in Sset:
                        return False
        return True

    if not is_valid(best):
        best = up
        best_sz = 64

    result = [int(i) for i in range(1, N + 1) if (best >> i) & 1]
    return result
