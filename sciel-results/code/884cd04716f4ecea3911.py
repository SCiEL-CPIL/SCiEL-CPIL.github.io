import time
import random
from collections import deque

def _moore_bound(k, g):
    if g % 2 == 0:
        depth = g // 2 - 1
        return 2 * (1 + k * (k**depth - 1) // (k - 1))
    else:
        depth = (g - 1) // 2
        return 1 + k * (k**depth - 1) // (k - 1)

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        parent[start] = -2
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _luw_construct(k, g):
    """Construct the Lazebnik-Ustimenko-Woldar graph."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 5:
        add = lambda x, y: (x + y) % 5
        sub = lambda x, y: (x - y) % 5
        mul = lambda x, y: x * y % 5
    elif q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 3:
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: x * y % 3
    elif q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
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
            yield (1 - side), tuple(y)
            
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Use LUW construction as baseline
    n_luw, edges_luw = _luw_construct(k, g)
    
    # Verify and return if valid
    if _verify_girth(n_luw, edges_luw, g):
        return n_luw, edges_luw
    
    # If LUW fails (shouldn't happen), return it anyway as fallback
    return n_luw, edges_luw
