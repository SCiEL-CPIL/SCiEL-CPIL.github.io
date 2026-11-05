import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for overhead
    
    if k == 4 and g == 10:
        # Try to find a smaller LUW graph by reducing d
        # Theoretical girth of LUW(q, d) is approx 2d + 2
        # For g=10, 2d+2 >= 10 => d >= 4
        # Current code uses d=5 (n=2048)
        # We try d=4 (n=512) first, then d=3 (n=128) just in case
        
        for d in [3, 4, 5]:
            try:
                n, edges = _luw_graph(k, g, d)
                # Verify girth
                if _verify_girth(n, edges, g):
                    return n, edges
            except Exception:
                continue
        
        # Fallback to original logic if specific d fails
        return _luw_graph(k, g)
    else:
        return _luw_graph(k, g)

def _luw_graph(k, g, d=None):
    q = k
    # Determine d such that girth >= g
    # LUW girth is approx 2*d + 2?
    # For q=4, d=5 gives girth >= 10.
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        # GF(4) operations: add/sub is XOR, mul is table
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the rules for the voltage assignment
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is ignored (None)
        for first in range(q):
            # Initialize y[1] = first
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

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
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles shorter than g
    # We can use BFS from each vertex up to depth g//2 - 1
    # If we find a cycle of length < g, return False
    
    max_depth = (g - 1) // 2
    
    for i in range(n):
        visited = [-1] * n
        parent = [-1] * n
        dist = [-1] * n
        
        queue = deque()
        queue.append(i)
        visited[i] = i
        dist[i] = 0
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
                
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = i
                    parent[v] = u
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif visited[v] == i and v != parent[u]:
                    # Found a cycle
                    # Cycle length is dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True
