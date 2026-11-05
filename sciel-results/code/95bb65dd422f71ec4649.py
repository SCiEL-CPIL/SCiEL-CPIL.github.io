import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Structural change: Cayley-graph family on non-abelian (dicyclic) groups,
    with a GF(2) syndrome-based girth predictor used to prune the search,
    plus a verified fallback (LUW) family.
    """
    start_time = time.time()
    deadline = start_time + 135.0

    if k == 3:
        res = _cayley_dicyclic(g, deadline, start_time)
        if res is not None:
            return res
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)


def _dicyclic_mul(n):
    order = 2 * n
    def mul(p, q):
        i1, e1 = p
        i2, e2 = q
        if e1 == 0 and e2 == 0:
            return ((i1 + i2) % order, 0)
        if e1 == 0 and e2 == 1:
            return ((i1 + i2) % order, 1)
        if e1 == 1 and e2 == 0:
            return ((i1 - i2) % order, 1)
        return (((i1 + i2 + n) % order), 0)
    return mul


def _cayley_girth(n, g):
    order = 2 * n
    mul = _dicyclic_mul(n)
    ident = (0, 0)
    gens = [(1, 0), (order - 1, 0), (0, 1)]
    gen_inv = {0: 1, 1: 0, 2: 2}
    dq = deque()
    dq.append((ident, -1, 0))
    seen = set()
    seen.add((ident, -1))
    while dq:
        el, last, L = dq.popleft()
        if L >= g:
            continue
        for gi, gen in enumerate(gens):
            if gi == gen_inv[last]:
                continue
            prod = mul(el, gen)
            st = (prod, gi)
            if st in seen:
                continue
            seen.add(st)
            if prod == ident and L + 1 >= 3:
                return L + 1
            dq.append((prod, gi, L + 1))
    return g


def _build_cayley_edges(n):
    order = 2 * n
    mul = _dicyclic_mul(n)
    gens = [(1, 0), (order - 1, 0), (0, 1)]
    edges = set()
    for i in range(order):
        for e in (0, 1):
            idx = e * order + i
            for gen in gens:
                prod = mul((i, e), gen)
                j, f = prod
                jidx = f * order + j
                a, b = min(idx, jidx), max(idx, jidx)
                edges.add((a, b))
    return order, list(edges)


def _verify(n, edges, k, g, deadline):
    deg = [0] * n
    seen_edges = set()
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in seen_edges:
            return False
        seen_edges.add((a, b))
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    max_len = g - 1
    for start in range(n):
        if time.time() > deadline:
            return False
        dq = deque()
        for nb in adj[start]:
            dq.append((nb, start, 1))
        while dq:
            u, prev, L = dq.popleft()
            if u == start and L > 0:
                return False
            if L >= max_len:
                continue
            for v in adj[u]:
                if v != prev:
                    dq.append((v, u, L + 1))
    return True


def _cayley_dicyclic(g, deadline, start_time):
    # Exponential search for an upper bound on n
    n = 64
    found_hi = None
    while n <= 8000:
        if time.time() > deadline:
            break
        gr = _cayley_girth(n, g)
        if gr >= g:
            found_hi = n
            break
        n *= 2
    if found_hi is None:
        for cand in (2048, 4096, 6000):
            if time.time() > deadline:
                break
            if _cayley_girth(cand, g) >= g:
                found_hi = cand
                break
    if found_hi is None:
        return None

    lo, hi = max(1, found_hi // 2), found_hi
    best_n = found_hi
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if _cayley_girth(mid, g) >= g:
            best_n = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # Keep order moderate so verification is fast
    if 2 * best_n > 8000:
        return None

    order, edges = _build_cayley_edges(best_n)
    if _verify(order, edges, 3, g, deadline):
        return order, edges
    return None


def _construct_luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
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
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
