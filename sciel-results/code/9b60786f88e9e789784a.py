import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _gf(q):
    if q == 4:
        t = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: t[x][y])
    return ((lambda x, y: (x + y) % q),
            (lambda x, y: (x - y) % q),
            (lambda x, y: x * y % q))


def _rule(d):
    r = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        m = (c - 5) % 4
        c0 = c - m
        r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][m]
    return r


def _luw_bipartite(q, d, deadline):
    add, sub, mul = _gf(q)
    rule = _rule(d)
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _luw_twist(q, d, deadline):
    add, sub, mul = _gf(q)
    rule = _rule(d)
    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
    n = len(order)
    adj = [[] for _ in range(n)]
    for vertex in order:
        side, x = vertex
        u = index[vertex]
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            v = index[(1 - side, tuple(y))]
            if v > u:
                adj[u].append(v)
                adj[v].append(u)
    return n, adj


def _girth_ok(adj, g):
    n = len(adj)
    for s in range(n):
        seen = {}
        dq = deque()
        for a in adj[s]:
            seen[a] = s
            dq.append((a, s, 1))
        while dq:
            u, p, depth = dq.popleft()
            if u == s and depth >= 1:
                return depth < g
            for w in adj[u]:
                if w == p:
                    continue
                if w in seen:
                    if seen[w] == u:
                        continue
                    return (depth + 1) < g
                seen[w] = u
                dq.append((w, u, depth + 1))
    return True


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 115.0
    q = k
    d = g - 2
    if d < 3:
        d = 3

    # Non-bipartite (twisted) LUW graph: same girth, ~half the vertices.
    try:
        n, adj = _luw_twist(q, d, deadline)
        if _girth_ok(adj, g):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
    except Exception:
        pass

    # Guaranteed-valid bipartite LUW graph (girth 2d+2 >= g).
    n, edges = _luw_bipartite(q, d, deadline)
    return n, edges

# EVOLVE-BLOCK-END
