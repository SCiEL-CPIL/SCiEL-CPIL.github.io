import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety

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

    def check_girth_nonbacktracking(n, edges, k, g):
        """Check girth >= g using non-backtracking BFS from each vertex."""
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            if u == v or u < 0 or v < 0 or u >= n or v >= n:
                return False
            if v in adj[u]:
                return False
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Non-backtracking BFS from each vertex
        for s in range(n):
            dq = deque()
            dist = {}
            for nb in adj[s]:
                dq.append((nb, s))
                dist[(nb, s)] = 1
            while dq:
                curr, prev = dq.popleft()
                depth = dist[(curr, prev)]
                if curr == s:
                    cycle_len = depth + 1
                    if cycle_len < g:
                        return False
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (nb, curr)
                    if state not in dist:
                        dist[state] = depth + 1
                        dq.append(state)
        return True

    # Generate LUW graph
    n_luw, edges_luw = luw_construct(k, g)
    
    # Verify it's valid
    if check_girth_nonbacktracking(n_luw, edges_luw, k, g):
        return n_luw, edges_luw
    else:
        # Should not happen, but return anyway
        return n_luw, edges_luw
