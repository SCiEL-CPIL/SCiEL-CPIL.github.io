import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0
    random.seed(12345)

    def elapsed():
        return time.time() - start_time

    def gen_random_3_regular(n):
        if n % 2 != 0:
            return None
        for _ in range(60):
            half = list(range(n)) * 3
            random.shuffle(half)
            adj = [[] for _ in range(n)]
            edge_set = set()
            ok = True
            for i in range(0, 3 * n, 2):
                u = half[i]; v = half[i + 1]
                if u == v:
                    ok = False; break
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in edge_set:
                    ok = False; break
                edge_set.add((a, b))
                adj[u].append(v); adj[v].append(u)
            if ok:
                return adj
        return None

    def _lift_neighbours(v, a, base_adj, voltages, m):
        for u in base_adj[v]:
            a2 = u if u < v else v
            t = voltages[(a2, u)]
            if v == a2:
                na = (a + t) % m
            else:
                na = (a - t) % m
            yield u * m + na

    def lift_girth(base_adj, voltages, m, g, n_base):
        total = n_base * m
        for s in range(total):
            sv = s // m; sa = s % m
            dist = {}
            dq = deque()
            for nb in _lift_neighbours(sv, sa, base_adj, voltages, m):
                dist[(s, nb)] = 1
                dq.append((s, nb))
            while dq:
                u, prev = dq.popleft()
                d = dist[(u, prev)]
                for w in _lift_neighbours(u // m, u % m, base_adj, voltages, m):
                    if w == prev:
                        continue
                    if w == s:
                        if d + 1 < g:
                            return d + 1
                    else:
                        key = (w, u)
                        if key in dist:
                            if dist[key] + d + 1 < g:
                                return dist[key] + d + 1
                        else:
                            dist[key] = d + 1
                            dq.append((w, u))
        return 10**9

    def assign_voltages(base_adj, n_base, m):
        voltages = {}
        for u in range(n_base):
            for v in base_adj[u]:
                if u < v:
                    voltages[(u, v)] = random.randrange(m)
        return voltages

    def build_lift_edges(base_adj, voltages, m, n_base):
        edges = []
        seen = set()
        for u in range(n_base):
            for v in base_adj[u]:
                if u < v:
                    t = voltages[(u, v)]
                    for a in range(m):
                        va = (a + t) % m
                        e1 = u * m + a
                        e2 = v * m + va
                        key = (e1, e2) if e1 < e2 else (e2, e1)
                        if key not in seen:
                            seen.add(key)
                            edges.append((e1, e2))
        return edges

    def verify_simple_3reg(edges, n):
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u < 0 or u >= n or v < 0 or v >= n or u == v:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                return False
            seen.add((a, b))
            adj[u].append(v); adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != 3:
                return False
        return True

    # Build base pool
    base_pool = []
    for nb in [10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60]:
        if elapsed() > deadline - 10:
            break
        adj = gen_random_3_regular(nb)
        if adj is not None:
            base_pool.append((nb, adj))

    if not base_pool:
        return _luw_fallback(k, g)

    m_options = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 32, 40, 48, 50, 60, 64, 80, 100, 120, 150]

    best = None
    for nb, base_adj in base_pool:
        if elapsed() > deadline - 5:
            break
        for m in m_options:
            if elapsed() > deadline - 5:
                break
            n = nb * m
            if best is not None and n >= best[0]:
                continue  # can't improve
            # Try several random voltage assignments
            for trial in range(40):
                if elapsed() > deadline - 5:
                    break
                voltages = assign_voltages(base_adj, nb, m)
                gl = lift_girth(base_adj, voltages, m, g, nb)
                if gl >= g:
                    edges = build_lift_edges(base_adj, voltages, m, nb)
                    if verify_simple_3reg(edges, n):
                        best = (n, edges)
                        break
            if best is not None and best[0] == n:
                break  # larger m only increases n for this nb

    if best is not None:
        return best
    return _luw_fallback(k, g)

def _luw_fallback(k, g):
    q = k
    d = 7
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x ^ y
        sub = lambda x,y: x ^ y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - (c-5) % 4
        rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)

    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
