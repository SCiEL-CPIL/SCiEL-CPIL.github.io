"""Direct constraint-satisfying 3-regular graph generator.

Builds the graph by pairing stubs one at a time (configuration model). A pairing is
accepted only if, in the current *partial* graph, no non-backtracking walk can close
the new edge into a cycle of length < g. This makes the girth condition hold by
construction (no post-hoc repair). A final exact certification is performed.
"""
# EVOLVE-BLOCK-START
import random
import sys
from collections import deque

sys.setrecursionlimit(1000000)

def _fast_girth_ok(adj, g, budget):
    """Return False as soon as a cycle of length < g is found; True if none found within budget."""
    n = len(adj)
    for s in range(n):
        # state = (vertex, prev_vertex, depth); depth = number of edges traversed
        dq = deque()
        for nb in adj[s]:
            dq.append((nb, s, 1))
        while dq:
            v, prev, d = dq.popleft()
            if d + 1 >= g:
                # extending would reach length >= g; cannot close a <g cycle
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s:
                    return False  # closed a cycle of length d+1 < g
                dq.append((nb, v, d + 1))
        budget[0] -= 1
        if budget[0] <= 0:
            return True
    return True

def _girth_ok(adj, g):
    """Exact: no cycle of length < g (certified via non-backtracking walks from every vertex)."""
    n = len(adj)
    for s in range(n):
        dq = deque()
        for nb in adj[s]:
            dq.append((nb, s, 1))
        while dq:
            v, prev, d = dq.popleft()
            if d + 1 >= g:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s:
                    return False
                dq.append((nb, v, d + 1))
    return True

def _build(k, g, seed):
    """Pair stubs with rejection sampling that forbids creating any <g cycle."""
    rng = random.Random(seed)
    n = 2 * k
    stubs = [i % k for i in range(2 * n)]
    rng.shuffle(stubs)
    adj = [[] for _ in range(n)]
    pairs = []
    for i in range(n):
        s = stubs[i]
        # find the earliest unused stub j
        j = i + 1
        while j < 2 * n and stubs[j] == s:
            j += 1
        if j >= 2 * n:
            return None  # should not happen
        if s == stubs[j]:
            return None  # loop
        # try random unused stubs (different vertex), reject if it creates a short cycle
        tried = 0
        max_tries = 400
        while tried < max_tries:
            t = rng.randrange(i + 1, 2 * n)
            if stubs[t] == s:
                tried += 1
                continue
            # tentatively add edge
            adj[s].append(stubs[t])
            adj[stubs[t]].append(s)
            budget = [n]
            ok = _fast_girth_ok(adj, g, budget)
            # undo
            adj[s].pop()
            adj[stubs[t]].pop()
            if ok:
                stubs[t] = s  # mark used
                pairs.append((s, stubs[t]))
                break
            tried += 1
        else:
            return None  # could not place a valid stub
    # verify degrees
    for v in range(n):
        if len(adj[v]) != k:
            return None
    if not _girth_ok(adj, g):
        return None
    return n, pairs

def construct(k, g):
    # small Moore-bound sanity: for k=3,g=16 the known optimum is 936; we build a
    # valid (not necessarily minimal) graph directly.
    best = None
    # try a handful of seeds; keep the first valid (all have same n for fixed k).
    for seed in range(0, 40):
        res = _build(k, g, seed)
        if res is not None:
            best = res
            break
    if best is None:
        # fallback: fall back to the explicit algebraic construction (still valid)
        return _lazebnik(k, g)
    return best

def _lazebnik(k, g):
    # Fallback explicit construction (Lazebnik-Ustimenko-Woldar style) to guarantee validity.
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
# EVOLVE-BLOCK-END
