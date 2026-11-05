import time
import random
import math
from collections import deque
import numpy as np


def _moore_bound(k, g):
    if g % 2 == 0:
        h = g // 2
        return 1 + k * sum((k - 1) ** i for i in range(h - 1))
    else:
        h = (g - 1) // 2
        return k * sum((k - 1) ** i for i in range(h))


def construct(k, g):
    # ---- LUW construction (fallback / general k,g) ----
    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
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

    if not (k == 4 and g == 10):
        return luw_construct(k, g)

    deadline = time.time() + 115.0
    
    # Try LUW d=4 (512 vertices) first, check girth
    # If girth >= 10, return it
    # Theoretical girth for even d is d+4? If so, 8. Likely invalid.
    # But let's check.
    n_luw4, edges_luw4 = luw_construct(4, 10, d_override=4)
    
    # Quick girth check for LUW4
    def get_girth(adj, limit=None):
        n = len(adj)
        if limit is None:
            limit = float('inf')
        best = float('inf')
        A = np.asarray(adj, dtype=np.int32)
        for s in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            dist[s] = 0
            frontier = np.array([s], dtype=np.int32)
            d = 0
            while frontier.size:
                d += 1
                if d >= best:
                    break
                nxt = A[frontier].ravel()
                nd = dist[nxt]
                cyc = nd[nd >= 0] + d + 1
                if cyc.size:
                    cm = int(cyc.min())
                    if cm < best:
                        best = cm
                        if best <= limit:
                            return best
                unv = nd < 0
                if unv.any():
                    newf = nxt[unv]
                    dist[newf] = d
                    frontier = np.unique(newf)
        return best

    # Build adj for LUW4
    adj_luw4 = [[] for _ in range(n_luw4)]
    for u, v in edges_luw4:
        adj_luw4[u].append(v)
        adj_luw4[v].append(u)
        
    girth_luw4 = get_girth(adj_luw4, limit=10)
    if girth_luw4 >= 10:
        return n_luw4, edges_luw4

    # Fallback to LUW d=5 (2048 vertices)
    n_luw5, edges_luw5 = luw_construct(4, 10, d_override=5)
    
    # Optional: Try random search for 320 vertices if time permits
    # But LUW5 is guaranteed valid.
    # Given the difficulty of finding 320-vertex cage, we return LUW5.
    return n_luw5, edges_luw5
