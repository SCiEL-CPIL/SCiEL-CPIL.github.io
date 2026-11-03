import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 150s kill
    rng = random.Random(20240517)

    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def cert_girth(n, adj, g):
        """True iff girth >= g, certified via non-backtracking BFS from every vertex."""
        for s in range(n):
            # dist[(v, prev)] = length of shortest non-backtracking walk from s ending at v
            # arriving from prev. A cycle of length L < g exists iff s is reachable with
            # such a walk of length L (L>=2).
            dist = {}
            q = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                q.append((nb, s, 1))
            while q:
                node, prev, d = q.popleft()
                if d >= g - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s:
                        # closed non-backtracking walk of length d+1 < g  -> short cycle
                        return False
                    key = (nb, node)
                    if key not in dist:
                        dist[key] = d + 1
                        q.append((nb, node, d + 1))
        return True

    def is_valid(n, edges, k, g):
        # simple + k-regular
        seen = set()
        deg = [0] * n
        for u, v in edges:
            if u == v or u < 0 or v < 0 or u >= n or v >= n:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                return False
            seen.add((a, b))
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != k:
                return False
        adj = build_adj(n, edges)
        return cert_girth(n, adj, g)

    def edges_from_adj(adj):
        e = []
        for u in range(len(adj)):
            for v in adj[u]:
                if u < v:
                    e.append((u, v))
        return e

    # ---- Guaranteed-valid fallback (fast, deterministic) ----
    best_n, best_edges = luw_construct(k, g)

    # ---- Portfolio search for a smaller valid graph (time-boxed) ----
    def try_candidate(n, edges):
        nonlocal best_n, best_edges
        if n >= best_n:
            return
        if time.time() > deadline:
            return
        if is_valid(n, edges, k, g):
            best_n, best_edges = n, edges

    def search_circulant(k, g, max_tries):
        # Moore bound lower bound
        r = (g - 1) // 2
        moore = 1 + sum(k * (k - 1) ** i for i in range(r))
        # n must satisfy: n*k even and k odd -> n even
        n_start = moore + 2
        if (n_start * k) % 2:
            n_start += 1
        n = n_start
        tried = 0
        while n < best_n and tried < max_tries and time.time() < deadline:
            tried += 1
            # choose k distinct generators up to sign
            half = n // 2
            if half < k:
                n += 2
                continue
            gen = rng.sample(range(1, half + 1), k)
            adj = [[] for _ in range(n)]
            ok = True
            for u in range(n):
                nbrs = []
                for a in gen:
                    nbrs.append((u + a) % n)
                    nbrs.append((u - a) % n)
                nbrs = list(dict.fromkeys(nbrs))
                if len(nbrs) != k:
                    ok = False
                    break
                adj[u] = nbrs
            if not ok:
                continue
            if is_valid(n, edges_from_adj(adj), k, g):
                best_n, best_edges = n, edges_from_adj(adj)
                return
            n += 2

    def random_regular(n, k, tries):
        # Configuration-model style: build k-regular simple graph, check girth.
        if (n * k) % 2:
            return None
        for _ in range(tries):
            if time.time() > deadline:
                return None
            stubs = []
            for u in range(n):
                for _ in range(k):
                    stubs.append(u)
            rng.shuffle(stubs)
            adj = [set() for _ in range(n)]
            ok = True
            for i in range(0, len(stubs), 2):
                a, b = stubs[i], stubs[i + 1]
                if a == b:
                    ok = False
                    break
                if b in adj[a]:
                    ok = False
                    break
                adj[a].add(b)
                adj[b].add(a)
            if not ok:
                continue
            adjl = [list(s) for s in adj]
            if all(len(s) == k for s in adjl):
                if is_valid(n, edges_from_adj(adjl), k, g):
                    return edges_from_adj(adjl)
        return None

    # Circulant search is cheap per trial and structurally girth-friendly; run first.
    if time.time() < deadline - 2:
        search_circulant(k, g, max_tries=20000)

    # Random-regular polish for k=3 (where small cages are reachable) if time remains.
    if k == 3 and best_n > 10 and time.time() < deadline - 2:
        # try a few n values near the Moore bound
        r = (g - 1) // 2
        moore = 1 + sum(k * (k - 1) ** i for i in range(r))
        for n in range(moore, best_n):
            if time.time() > deadline - 2:
                break
            if (n * k) % 2:
                continue
            e = random_regular(n, k, tries=200)
            if e is not None:
                best_n, best_edges = n, e
                break

    return best_n, best_edges
