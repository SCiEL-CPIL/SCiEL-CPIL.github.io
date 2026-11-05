import time
import numpy as np
from collections import deque

_T0 = time.time()
_DEADLINE = _T0 + 110.0

def _deadline():
    return time.time() >= _DEADLINE

def _field_ops(q):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    return ((lambda x, y: (x + y) % q),
            (lambda x, y: (x - y) % q),
            (lambda x, y: (x * y) % q))

def _luw_rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule

def _luw_graph(d, q):
    add, sub, mul = _field_ops(q)
    rule = _luw_rule(d)
    rc = [None] * (d + 1)
    for c in range(2, d + 1):
        rc[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rc[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
    
    adj = [set() for _ in range(len(order))]
    for vertex in order:
        i = index[vertex]
        for other in neighbours(vertex):
            j = index[other]
            adj[i].add(j)
            
    edges = [(i, j) for i in range(len(order)) for j in adj[i] if i < j]
    return len(order), edges, adj

def _verify_girth(adj, g):
    """Verify girth is at least g. Returns True if valid."""
    n = len(adj)
    if n == 0:
        return True
    
    # Use non-backtracking BFS from each vertex
    # For efficiency, we can early exit if we find a short cycle
    for s in range(n):
        if _deadline():
            return False
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= (g - 1) // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and v != s:
                    # Potential cycle
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
                elif v == s:
                    cycle_len = du + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    best = None
    best_n = float('inf')
    
    # LUW construction: girth = d + 5, degree = q
    # For girth g, degree k: d = g - 5, q = k
    d = g - 5
    if d < 2:
        d = 2
    
    # Try the primary LUW construction
    try:
        n, edges, adj = _luw_graph(d, k)
        
        # Verify regularity
        deg = [0] * n
        for (i, j) in edges:
            deg[i] += 1
            deg[j] += 1
        
        if all(x == k for x in deg):
            # Verify girth
            if _verify_girth(adj, g):
                best_n = n
                best = (n, edges)
    except Exception:
        pass
    
    # If primary didn't work, try variations
    if best is None:
        for d_try in range(max(2, d - 2), d + 3):
            if _deadline():
                break
            try:
                n, edges, adj = _luw_graph(d_try, k)
                deg = [0] * n
                for (i, j) in edges:
                    deg[i] += 1
                    deg[j] += 1
                if all(x == k for x in deg):
                    if _verify_girth(adj, g):
                        if n < best_n:
                            best_n = n
                            best = (n, edges)
            except Exception:
                continue
    
    if best is not None:
        return best
    
    # Fallback
    return 0, []
