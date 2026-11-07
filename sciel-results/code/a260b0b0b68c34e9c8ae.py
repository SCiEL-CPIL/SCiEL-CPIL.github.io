"""Small exact biclique partitions of a random graph."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    t0 = time.time()
    deadline = t0 + 78.0
    n = P['n']
    edges = P['edges']
    Eset = set()
    for a, b in edges:
        if a < b:
            Eset.add((a, b))
        else:
            Eset.add((b, a))
    E = sorted(Eset)
    nE = len(E)
    adj = [set() for _ in range(n)]
    for u, v in E:
        adj[u].add(v)
        adj[v].add(u)

    def cross_ok(L, R, eset):
        for u in L:
            for v in R:
                if (min(u, v), max(u, v)) not in eset:
                    return False
        return True

    def edges_of(L, R):
        s = set()
        for u in L:
            for v in R:
                s.add((min(u, v), max(u, v)))
        return s

    def valid_partition(bcs):
        if not bcs:
            return nE == 0
        seen = set()
        for L, R in bcs:
            if not L or not R:
                return False
            if set(L) & set(R):
                return False
            if not cross_ok(L, R, Eset):
                return False
            for e in edges_of(L, R):
                if e in seen:
                    return False
                seen.add(e)
        return seen == Eset

    def greedy_seed():
        remset = set(E)
        bcs = []
        while remset:
            if time.time() > deadline:
                break
            best = None
            best_cov = -1
            for (u, v) in remset:
                cand = set()
                for x in adj[u]:
                    if x == v:
                        continue
                    if (min(u, x), max(u, x)) in remset:
                        cand.add(x)
                cand.add(u)
                R = set()
                for y in adj[v]:
                    if y == u:
                        continue
                    ok = True
                    for x in cand:
                        if (min(x, y), max(x, y)) not in remset:
                            ok = False
                            break
                    if ok:
                        R.add(y)
                R.add(v)
                cand -= R
                if not cand or not R:
                    continue
                cov = 0
                for x in cand:
                    for y in R:
                        if (min(x, y), max(x, y)) in remset:
                            cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best = (list(cand), list(R))
            if best is None or best_cov <= 0:
                u, v = next(iter(remset))
                best = ([u], [v])
            L, R = best
            bcs.append((L, R))
            new_rem = set()
            for e in remset:
                a, b = e
                if (a in L and b in R) or (a in R and b in L):
                    continue
                new_rem.add(e)
            remset = new_rem
        return bcs

    def try_merge(bcs, i, j):
        L1, R1 = bcs[i]
        L2, R2 = bcs[j]
        s1, s2 = set(L1), set(R1)
        t1, t2 = set(L2), set(R2)
        if s1 & t1 or s2 & t2 or s1 & t2 or s2 & t1:
            return None
        L = s1 | t1
        R = s2 | t2
        if not cross_ok(L, R, Eset):
            return None
        e1 = edges_of(L1, R1)
        e2 = edges_of(L2, R2)
        if e1 & e2:
            return None
        if edges_of(L, R) != (e1 | e2):
            return None
        return (list(L), list(R))

    def local_search(bcs, rng):
        bcs = [list(x) for x in bcs]
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            nB = len(bcs)
            order = list(range(nB))
            rng.shuffle(order)
            for i in order:
                for j in order:
                    if i >= j:
                        continue
                    if time.time() > deadline:
                        break
                    m = try_merge(bcs, i, j)
                    if m is not None:
                        newbcs = [bcs[k] for k in range(nB) if k != i and k != j]
                        newbcs.append(m)
                        bcs = newbcs
                        improved = True
                        break
                if improved:
                    break
        return bcs

    def random_split(bcs, rng):
        bcs = [list(x) for x in bcs]
        if not bcs:
            return bcs
        idx = rng.randrange(len(bcs))
        L, R = bcs[idx]
        if len(L) >= 2:
            Lc = list(L)
            rng.shuffle(Lc)
            k = rng.randrange(1, len(Lc))
            L1, L2 = Lc[:k], Lc[k:]
            if cross_ok(L1, R, Eset) and cross_ok(L2, R, Eset):
                bcs[idx] = (L1, list(R))
                bcs.append((L2, list(R)))
                return bcs
        if len(R) >= 2:
            Rc = list(R)
            rng.shuffle(Rc)
            k = rng.randrange(1, len(Rc))
            R1, R2 = Rc[:k], Rc[k:]
            if cross_ok(L, R1, Eset) and cross_ok(L, R2, Eset):
                bcs[idx] = (list(L), R1)
                bcs.append((list(L), R2))
                return bcs
        return bcs

    rng = random.Random(12345)
    bcs = greedy_seed()
    if not valid_partition(bcs):
        bcs = []
        for u in range(n):
            R = [v for (a, b) in E for v in (b if a == u else a) if a == u]
            if R:
                bcs.append(([u], R))
    best = [list(x) for x in bcs]
    best_len = len(best)

    iters = 0
    while time.time() < deadline - 2.0:
        iters += 1
        cur = [list(x) for x in best]
        if iters % 3 == 0:
            cur = random_split(cur, rng)
            if not valid_partition(cur):
                cur = [list(x) for x in best]
        cur = local_search(cur, rng)
        if valid_partition(cur) and len(cur) < best_len:
            best = cur
            best_len = len(cur)
        if best_len == 1:
            break

    if time.time() < deadline - 1.0:
        best = local_search(best, rng)

    out = []
    for L, R in best:
        out.append([[int(x) for x in L], [int(x) for x in R]])
    return out
# EVOLVE-BLOCK-END
