import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55

    N = 512
    LIMIT = 512

    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        m = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= LIMIT:
                m |= (1 << (j - 1))
        adj[i] = m

    full = (1 << N) - 1

    def is_clique(S):
        # S is bitmask
        if S == 0:
            return True
        temp = S
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            if (S & ~adj[v]) != 0:
                return False
            temp ^= lsb
        return True

    def greedy_by_degree():
        P = full
        R = []
        remaining = full
        while remaining:
            best = -1
            best_deg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                d = (P & adj[v]).bit_count()
                if d > best_deg:
                    best_deg = d
                    best = v
                temp ^= lsb
            R.append(best)
            P &= adj[best]
            remaining &= ~ (1 << (best - 1))
            remaining &= P
        return R

    best = greedy_by_degree()
    best_mask = 0
    for v in best:
        best_mask |= (1 << (v - 1))
    best_size = len(best)

    def color_bound(P):
        if P == 0:
            return 0
        colors = 0
        uncolored = P
        while uncolored:
            colors += 1
            indep = 0
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                if (indep & adj[v]) == 0:
                    indep |= lsb
                    uncolored ^= lsb
                temp ^= lsb
        return colors

    import sys
    sys.setrecursionlimit(100000)

    def bk(R, P, X):
        nonlocal best, best_size, best_mask
        if time.time() > deadline:
            return
        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best = R[:]
                m = 0
                for v in R:
                    m |= (1 << (v - 1))
                best_mask = m
            return
        if len(R) + P.bit_count() <= best_size:
            return
        cb = color_bound(P)
        if len(R) + cb <= best_size:
            return
        PX = P | X
        if PX == 0:
            return
        best_ov = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length()
            ov = (P & adj[u]).bit_count()
            if ov > best_ov:
                best_ov = ov
                pivot = u
            temp ^= lsb
        if pivot == -1:
            return
        cand = P & ~adj[pivot]
        temp = cand
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            temp ^= lsb
            bk(R + [v], P & adj[v], X & adj[v])
            P &= ~lsb
            X |= lsb
            if time.time() > deadline:
                return

    bk([], full, 0)

    # Local search polish with bitmasks
    if time.time() < deadline - 2:
        cur = best_mask
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # try add
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if cur & bit:
                    continue
                if (cur & ~adj[v]) == 0:
                    cur |= bit
                    improved = True
                    break
            if improved:
                continue
            # try swap: remove one, add one (net same size) won't help; try remove one add two
            # Instead: find a vertex v not in cur, and a vertex u in cur such that (cur - u + v) is clique and we can then add more
            # Simpler: for each v not in cur, find u in cur to remove so that (cur-u)+v is clique; if so do it and continue (may enable adds)
            for v in range(1, N + 1):
                bit_v = 1 << (v - 1)
                if cur & bit_v:
                    continue
                # need to remove a set of cur vertices that conflict with v
                conflicts = cur & ~adj[v]
                if conflicts == 0:
                    cur |= bit_v
                    improved = True
                    break
                # try removing one conflicting vertex
                temp = conflicts
                found = False
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length()
                    newcur = (cur & ~lsb) | bit_v
                    # check newcur is clique
                    if is_clique(newcur):
                        cur = newcur
                        improved = True
                        found = True
                        break
                    temp ^= lsb
                if found:
                    break
        if cur.bit_count() > best_size:
            best_size = cur.bit_count()
            best = [v for v in range(1, N + 1) if cur & (1 << (v - 1))]
            best_mask = cur

    return best
