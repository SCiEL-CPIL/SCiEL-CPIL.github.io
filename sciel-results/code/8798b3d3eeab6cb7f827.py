import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110

    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2
            return k * (k - 1) ** (h - 1)
        else:
            h = (g - 1) // 2
            return 1 + k * sum((k - 1) ** i for i in range(h))

    def luw_construct(q, d):
        """
        Lazebnik-Ustimenko-Woldar construction.
        q: prime power
        d: depth parameter
        Returns (n, edges) for a 3-regular graph.
        """
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

        rule = {2: (1, 1), 3: (2, 1)}
        for c in range(4, d + 1):
            r = (c - 4) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y))

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []

        for vi, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    def girth_check_optimized(n, adj, max_girth):
        """
        Optimized girth check: for each vertex, enumerate non-backtracking walks
        up to length max_girth-1 and check if any returns to the start.
        """
        for start in range(n):
            dist = {}
            queue = deque()

            for nb in adj[start]:
                queue.append((nb, start, 1))
                dist[(nb, start)] = 1

            while queue:
                u, prev, d = queue.popleft()

                if d >= max_girth:
                    continue

                for nb in adj[u]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return False
                    if (nb, u) not in dist:
                        dist[(nb, u)] = d + 1
                        queue.append((nb, u, d + 1))

        return True

    def random_3reg(n):
        for _ in range(200):
            stubs = []
            for i in range(n):
                stubs += [i] * 3
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            ok = True
            for i in range(0, 3 * n, 2):
                a, b = stubs[i], stubs[i + 1]
                if a == b or b in adj[a]:
                    ok = False
                    break
                adj[a].add(b)
                adj[b].add(a)
            if ok:
                return [list(s) for s in adj]
        return None

    # Strategy: Try LUW with small q and d values
    # LUW(q, d) gives girth approximately 2*d + 2
    # For girth 16, we need d >= 7
    candidates = []
    for q in [3, 5, 7]:
        for d in range(5, 12):
            if time.time() > deadline - 20:
                break
            try:
                n, edges = luw_construct(q, d)
                if n <= 10000:
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    has_girth = girth_check_optimized(n, adj, g)
                    candidates.append((n, q, d, edges, has_girth))
            except Exception:
                continue

    valid = [(n, q, d, edges) for n, q, d, edges, hg in candidates if hg]
    if valid:
        valid.sort(key=lambda x: x[0])
        n, q, d, edges = valid[0]
        return n, edges

    # Fallback: random 3-regular graphs
    for n_try in range(512, 1000, 32):
        if time.time() > deadline - 15:
            break
        if n_try % 2 != 0:
            continue
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            adj = random_3reg(n_try)
            if adj is None:
                continue
            if girth_check_optimized(n_try, adj, g):
                edges = []
                for i in range(n_try):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n_try, edges

    # Final fallback: LUW with q=3, d=10
    n, edges = luw_construct(3, 10)
    return n, edges
