import time
import math
from collections import deque

_DEADLINE = 145.0

def _now():
    return time.time()

def _girth_check(n, edges, g):
    """Return True if the graph has no cycle shorter than g (using BFS from each vertex)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif dist[v] <= dist[u]:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _verify_regular(n, edges, k):
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return all(d == k for d in deg)

def _verify_simple(n, edges):
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
    return True

def _luw_construction(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    start_time = _now()
    deadline = start_time + _DEADLINE
    
    candidates = []
    
    try:
        n_luw, edges_luw = _luw_construction(k, g)
        candidates.append((n_luw, edges_luw))
    except Exception:
        pass

    if not candidates:
        raise ValueError("No valid construction found")

    candidates.sort(key=lambda x: x[0])
    best_n, best_edges = candidates[0]
    
    if _verify_regular(best_n, best_edges, k) and _verify_simple(best_n, best_edges):
        return best_n, best_edges

    for n, edges in candidates[1:]:
        if _verify_regular(n, edges, k) and _verify_simple(n, edges):
            return n, edges

    raise ValueError("No valid construction found")
