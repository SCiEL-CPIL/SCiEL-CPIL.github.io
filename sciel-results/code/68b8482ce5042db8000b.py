import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # For k=5, g=9, the LUW construction with d=4 yields n=1250.
    # This is a known valid construction for girth 9.
    # The seed program might have used d=5 (n=6250) if it incorrectly rejected d=4.
    # We explicitly try d=4 first.
    
    if k == 5 and g == 9:
        # Try d=4 (n=1250)
        n, edges = _luw_construct(5, 4)
        if _verify_girth(n, edges, 9):
            return n, edges
        
        # Fallback to d=5 (n=6250) if d=4 fails (should not happen)
        n, edges = _luw_construct(5, 5)
        if _verify_girth(n, edges, 9):
            return n, edges

    # General fallback
    min_d = (g - 1) // 2
    if (g - 1) % 2 != 0:
        min_d += 1

    for d in range(min_d, min_d + 5):
        if time.time() > deadline - 10:
            break
        n, edges = _luw_construct(k, d)
        if _verify_girth(n, edges, g):
            return n, edges

    d_fallback = min_d + 5
    return _luw_construct(k, d_fallback)

def _luw_construct(k, d):
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

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

def _verify_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
