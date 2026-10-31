"""EVOLVE-BLOCK: LUW base graph + girth-preserving voltage-lift search + fast girth check."""
from collections import deque
import time

def _field_ops(q):
    if q == 4:
        t = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: t[x][y])
    return ((lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: (x * y) % q))

def _luw_adj(q, d):
    add, sub, mul = _field_ops(q)
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for v in order:
        for o in neighbours(v):
            if o not in index:
                index[o] = len(order)
                order.append(o)
    n = len(order)
    adj = [[] for _ in range(n)]
    for v in order:
        side, x = v
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            adj[index[v]].append(index[tuple((1 - side, tuple(y)))])
    return adj

def _has_short_cycle(adj, k, g):
    n = len(adj)
    INF = 10 ** 9
    dist = [[INF] * k for _ in range(n)]
    for v in range(n):
        dist[v][v] = 0
    dq = deque()
    for v in range(n):
        for u in adj[v]:
            dq.append((u, v, 1))
    while dq:
        cur, prev, d = dq.popleft()
        if d >= g - 1:
            continue
        for nb in adj[cur]:
            if nb == prev:
                continue
            if dist[nb][cur] < INF:
                if dist[nb][cur] + d + 1 < g:
                    return True
            if d + 1 < dist[cur][nb] and d + 1 < g - 1:
                dist[cur][nb] = d + 1
                dq.append((nb, cur, d + 1))
    return False

def _lift(adj, m, volt):
    n = len(adj)
    adj2 = [[] for _ in range(n * m)]
    for u in range(n):
        for v in adj[u]:
            a = u * m + (volt[u][v] % m)
            b = v * m
            adj2[a].append(b)
            adj2[b].append(a)
    return adj2

def _lift_edges(adj, m, volt):
    n = len(adj)
    edges = []
    for u in range(n):
        for v in adj[u]:
            a = u * m + (volt[u][v] % m)
            b = v * m
            edges.append((a, b) if a < b else (b, a))
    return edges

def construct(k, g):
    t0 = time.time()
    DEAD = 145.0
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    base_adj = _luw_adj(q, d)
    nbase = len(base_adj)
    best = (nbase, _lift_edges(base_adj, 1, [[0] * k for _ in range(nbase)]))

    def elapsed():
        return time.time() - t0

    # Voltage-lift search: for small m, try several random voltage assignments.
    # A lift is valid iff the lifted graph has no cycle < g.
    for m in range(2, 12):
        if elapsed() > DEAD - 10.0:
            break
        trials = 40
        found = False
        for trial in range(trials):
            if elapsed() > DEAD - 10.0:
                break
            volt = [[0] * k for _ in range(nbase)]
            seed = 1000003 * m + 7 * trial + 13
            for u in range(nbase):
                for v in base_adj[u]:
                    seed = (seed * 1103515245 + 12345) & 0x7fffffff
                    volt[u][v] = (seed >> 8) % m
            adj2 = _lift(base_adj, m, volt)
            if not _has_short_cycle(adj2, k, g):
                cand = (nbase * m, _lift_edges(base_adj, m, volt))
                if cand[0] < best[0]:
                    best = cand
                found = True
                break
        if found:
            break
    return best
