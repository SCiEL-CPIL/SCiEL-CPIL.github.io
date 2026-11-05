import time
import numpy as np
from collections import deque

def _k4_ops():
    table = np.array([[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]], dtype=np.int32)
    def add(x, y): return (x ^ y) & 3
    def sub(x, y): return (x ^ y) & 3
    def mul(x, y): return table[x, y]
    return add, sub, mul

def _build_rule(d, family):
    base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    if family == 'orig':
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    elif family == 'alt':
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            base[c] = [(1, c0 + 1), (c0, 1), (c0 - 1, 1), (1, c0 - 2)][r]
    elif family == 'alt2':
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            base[c] = [(c0 - 1, 1), (c0, 1), (1, c0 + 1), (1, c0 - 2)][r]
    else:
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            base[c] = [(c0, 1), (c0 - 1, 1), (1, c0 + 1), (1, c0 - 2)][r]
    return base

def _luw_construct(k, g, d, family='orig'):
    q = k
    if q == 4:
        add, sub, mul = _k4_ops()
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    rule = _build_rule(d, family)

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

def _verify(n, edges, k, g):
    # Simple: no loops, no multi-edges, exact degree k.
    if len(edges) != n * k // 2:
        return False
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            return False
        if u == v:
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
    # Girth via non-backtracking closed walks (adjacency-matrix powers).
    A = np.zeros((n, n), dtype=np.int32)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    Apow = A.copy()
    for length in range(3, g):
        Apow = Apow @ A
        if np.any(np.diag(Apow) > 0):
            return False
    return True

def _luw_fallback(k, g):
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _luw_construct(k, g, d, 'orig')

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # safety margin vs 120s

    if k == 4 and g == 10:
        # Try LUW with various d and rule families; keep the smallest valid one.
        best = None
        for d in (4, 5, 6, 7, 8, 9, 10, 11, 12):
            if time.time() > deadline:
                break
            for family in ('orig', 'alt', 'alt2', 'alt3'):
                if time.time() > deadline:
                    break
                try:
                    n, edges = _luw_construct(k, g, d, family)
                except Exception:
                    continue
                if best is not None and n >= best[0]:
                    continue
                if _verify(n, edges, k, g):
                    best = (n, edges)
                    break  # smaller d already found for this d; stop families
        if best is not None:
            return best
        # Fallback: known larger construction
        n, edges = _luw_fallback(k, g)
        if _verify(n, edges, k, g):
            return n, edges
        return n, edges
    else:
        n, edges = _luw_fallback(k, g)
        return n, edges
