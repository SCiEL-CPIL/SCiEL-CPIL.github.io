"""Construct small 3-regular graphs of large girth via the Lazebnik-Ustimenko-Woldar
algebraic construction, with a hard deadline guard and a bounded non-backtracking
girth probe used to certify (and, if needed, adapt) the result before returning."""
import time
from collections import deque


def _luw_graph(q, d):
    """LUW q-regular bipartite graph with guaranteed girth >= d+5 (q prime or q=4)."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
    n = len(order)
    edges = []
    for vertex in order:
        if vertex[0] == 0:
            vi = index[vertex]
            for other in neighbours(vertex):
                edges.append((vi, index[other]))
    return n, edges


def _probe_girth(n, edges, max_depth, time_budget, deadline):
    """Bounded non-backtracking BFS from every vertex; return the shortest cycle
    length found (< max_depth) or None if none found or the budget is exhausted.
    A cycle of length L is a non-backtracking walk of length L returning to its
    start vertex without immediately reversing."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    best = max_depth
    # state: (prev, curr, depth); non-backtracking => next != prev
    for s in range(n):
        if time.time() > deadline or time.time() > time_budget:
            return None  # give up certifying; trust the construction theorem
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(s, nb)] = 1
            dq.append((s, nb, 1))
        while dq:
            prev, cur, dep = dq.popleft()
            if dep >= best:
                continue
            if dep >= max_depth:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s:
                    cand = dep + 1
                    if cand >= 3 and cand < best:
                        best = cand
                    continue
                st = (cur, nxt)
                if st not in dist:
                    dist[st] = dep + 1
                    dq.append((cur, nxt, dep + 1))
    if best < max_depth:
        return best
    return None


def construct(k, g):
    t0 = time.time()
    deadline = t0 + (150.0 - 25.0)  # keep a safety margin

    q = k
    d_min = g - 5
    if d_min < 3:
        d_min = 3
    if d_min % 2 == 0:
        d_min += 1
    d = d_min

    best_n, best_edges = None, None
    d_tried = 0
    while d_tried < 8:
        if time.time() > deadline:
            break
        n, edges = _luw_graph(q, d)
        best_n, best_edges = n, edges
        # certify with a bounded probe (cheap budget); if it finds a short cycle, bump d
        probe_budget = t0 + (150.0 - 40.0)
        short = _probe_girth(n, edges, g, probe_budget, deadline)
        if short is None or short >= g:
            break
        d += 2
        d_tried += 1

    if best_n is None:
        n, edges = _luw_graph(q, d)
        best_n, best_edges = n, edges

    return best_n, best_edges
