import time
import random
import numpy as np

def construct(k, g):
    t0 = time.time()
    DEADLINE = t0 + 118.0

    def rem():
        return DEADLINE - time.time()

    # ---------- fast girth check (BFS from every vertex) ----------
    def check_girth(n, edges, min_girth, limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j); adj[j].append(i)
        best = 1 << 30
        for s in range(n):
            if time.time() > limit:
                return -1
            dist = [-1] * n
            par = [-1] * n
            dist[s] = 0
            q = [s]; qi = 0
            while qi < len(q):
                node = q[qi]; qi += 1
                nd = dist[node] + 1
                for nb in adj[node]:
                    if nb == par[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = nd
                        par[nb] = node
                        q.append(nb)
                    else:
                        c = dist[node] + dist[nb] + 1
                        if c < best:
                            best = c
                        if c < min_girth:
                            return c
        return best

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None
        pet = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
               (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
               (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
        n_base = 10
        eidx = {}
        for i, (u, v) in enumerate(pet):
            eidx[(u, v)] = i
            eidx[(v, u)] = i
        adj = [[] for _ in range(n_base)]
        for u, v in pet:
            adj[u].append(v); adj[v].append(u)

        # all non-backtracking closed walks of length 5..15 -> net edge counts
        maxL = g - 1
        vecs = set()
        for s in range(n_base):
            stack = [(s, -1, 0, [0] * 15)]
            while stack:
                cur, prev, L, counts = stack.pop()
                if L >= maxL:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    ei = eidx[(cur, nxt)]
                    u, v = pet[ei]
                    d = 1 if cur == u else -1
                    nc = counts.copy(); nc[ei] += d
                    nl = L + 1
                    if nxt == s and nl >= 5:
                        vecs.add(tuple(nc))
                    stack.append((nxt, cur, nl, nc))

        uniq = set()
        for c in vecs:
            if all(x == 0 for x in c):
                continue
            f = -1
            for i in range(15):
                if c[i] != 0:
                    f = i; break
            if f != -1 and c[f] < 0:
                c = tuple(-x for x in c)
            uniq.add(c)
        C = np.array(list(uniq), dtype=np.int64)
        if C.size == 0:
            return None
        Cn = C.shape[0]

        def solve(m):
            Cmod = C % m
            v = np.random.randint(1, m, size=15).astype(np.int64)
            res = (Cmod @ v) % m
            best = int(np.sum(res == 0))
            if best == 0:
                return v
            T = 1.0
            cool = 0.9999
            for it in range(20000):
                if rem() < 6:
                    break
                r = random.random()
                if r < 0.5:
                    j = random.randint(0, 14)
                    ov = v[j]; v[j] = random.randint(1, m - 1)
                    res = (Cmod @ v) % m
                    nc = int(np.sum(res == 0))
                    if nc < best or (T > 0.02 and random.random() < np.exp(-(nc - best) / T)):
                        if nc < best:
                            best = nc; v0 = v.copy()
                        if nc == 0:
                            return v
                    else:
                        v[j] = ov
                else:
                    j1 = random.randint(0, 14); j2 = random.randint(0, 14)
                    if j1 == j2:
                        continue
                    o1, o2 = v[j1], v[j2]
                    v[j1] = o2; v[j2] = o1
                    res = (Cmod @ v) % m
                    nc = int(np.sum(res == 0))
                    if nc < best or (T > 0.02 and random.random() < np.exp(-(nc - best) / T)):
                        if nc < best:
                            best = nc; v0 = v.copy()
                        if nc == 0:
                            return v
                    else:
                        v[j1] = o1; v[j2] = o2
                T *= cool
            return None

        def build(v):
            n = 10 * m
            edges = []
            for u, vb in pet:
                vol = v[eidx[(u, vb)]]
                for layer in range(m):
                    a = u * m + layer
                    b = vb * m + (layer + vol) % m
                    edges.append((a, b) if a < b else (b, a))
            return n, edges

        m = 53
        while m < 1500 and rem() > 25:
            if is_prime(m):
                for _ in range(30):
                    if rem() < 12:
                        break
                    v = solve(m)
                    if v is not None:
                        n, edges = build(v)
                        gc = check_girth(n, edges, g, DEADLINE - 4)
                        if gc >= g:
                            return (n, edges)
            m += 1
        return None

    def luw_fallback(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if k == 3 and g == 16:
        r = voltage_lift_petersen()
        if r is not None:
            return r
    return luw_fallback(k, g)
