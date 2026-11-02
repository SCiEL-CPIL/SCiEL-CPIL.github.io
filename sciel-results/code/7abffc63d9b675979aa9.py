import time
import math
from math import gcd
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 120
    
    # Moore bound for k=3, g=16 is 766
    # We search for circulant graphs C(n; {a, -a, n/2})
    # n must be even.
    
    start_n = 766
    if start_n % 2 != 0:
        start_n += 1
        
    max_n = 2000 # Limit search to keep within time budget
    
    for n in range(start_n, max_n, 2):
        if time.time() > deadline - 10:
            break
            
        half_n = n // 2
        # a must be coprime to n/2 for connectivity
        for a in range(1, half_n):
            if time.time() > deadline - 5:
                break
            if gcd(a, half_n) != 1:
                continue
                
            # Check girth using BFS
            if _has_short_cycle_bfs(n, a, g):
                continue
            
            # Construct edges
            edges = []
            step1 = a
            step2 = -a % n
            step3 = half_n
            
            for i in range(n):
                j1 = (i + step1) % n
                j2 = (i + step2) % n
                j3 = (i + step3) % n
                
                if i < j1:
                    edges.append((i, j1))
                if i < j2:
                    edges.append((i, j2))
                if i < j3:
                    edges.append((i, j3))
            
            return n, edges

    # Fallback to LUW if no circulant found
    return _luw_fallback(k, g)

def _has_short_cycle_bfs(n, a, g):
    # BFS to find shortest cycle
    # If shortest cycle < g, return True
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    queue = deque([0])
    
    steps = (a, n - a, n // 2)
    
    while queue:
        u = queue.popleft()
        for s in steps:
            v = (u + s) % n
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
            elif v != parent[u]:
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g:
                    return True
    return False

def _luw_fallback(k, g):
    # Original LUW implementation
    deadline = time.time() + 120
    q = k
    # Optimize d: we need girth >= g. LUW gives girth >= 2d+2.
    # So 2d+2 >= g => d >= (g-2)/2.
    d = (g - 2 + 1) // 2  # ceil((g-2)/2)
    if d < 3:
        d = 3
        
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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
