import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110
    best_n = float('inf')
    best_edges = None
    
    # Try voltage lift construction
    n, edges = _voltage_lift_construct(k, g, deadline)
    if n is not None and n < best_n:
        best_n, best_edges = n, edges
    
    # Try LUTW as fallback
    if time.time() < deadline:
        n2, e2 = _lutw(k, g)
        if n2 < best_n:
            best_n, best_edges = n2, e2
    
    return best_n, best_edges

def _voltage_lift_construct(k, g, deadline):
    # Use K_5 as base graph, lift with Z_64
    base_vertices = 5
    base_edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
    m = 64
    
    for attempt in range(500):
        if time.time() > deadline:
            return None, None
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m-1)
        
        n, edges = _build_lift(base_edges, m, voltages)
        if _check_girth(n, edges, g):
            return n, edges
    
    return None, None

def _build_lift(base_edges, m, voltages):
    n = 5 * m
    edges = []
    for u, v in base_edges:
        volt = voltages[(u, v)]
        for g in range(m):
            u_lift = u * m + g
            v_lift = v * m + (g + volt) % m
            edges.append((u_lift, v_lift))
    return n, edges

def _check_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        visited = {start: 0}
        queue = deque([(start, 0, -1)])
        while queue:
            v, depth, parent = queue.popleft()
            if depth >= (g-1)//2:
                continue
            for w in adj[v]:
                if w == parent:
                    continue
                if w in visited:
                    if depth + 1 + visited[w] < g:
                        return False
                else:
                    visited[w] = depth + 1
                    queue.append((w, depth + 1, v))
    return True

def _lutw(k, g):
    # Same as original
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
