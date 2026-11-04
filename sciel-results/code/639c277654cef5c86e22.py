import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    if k == 4 and g == 10:
        # Try circulant graphs C(n, {a, -a, b, -b}) for n from 243 to 400
        for n_attempt in range(243, 400, 1):
            if time.time() > deadline - 15:
                break
            try:
                n, edges = _circulant_construct(n_attempt, 4, 10)
                if n is not None:
                    return n, edges
            except Exception:
                pass
        
        # If circulant failed, fall back to LUW
        return _luw_graph(k, g)
    else:
        # For other parameters, use LUW as a safe fallback
        return _luw_graph(k, g)

def _luw_graph(k, g):
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

def _circulant_construct(n, k, g):
    """Try circulant graphs C(n, {a, -a, b, -b}) for k=4."""
    if n < 5:
        return None, None
    
    best_girth = float('inf')
    best_params = None
    
    for a in range(1, n // 2 + 1):
        for b in range(a + 1, n // 2 + 1):
            # Check that {a, n-a, b, n-b} are all distinct mod n
            gens = {a % n, (-a) % n, b % n, (-b) % n}
            if len(gens) < 4:
                continue
            # Also ensure no generator is 0
            if 0 in gens:
                continue
            
            # Compute girth
            girth = _circulant_girth(n, a, b)
            if girth >= g:
                # Found a valid graph
                edges = []
                for i in range(n):
                    for d in [a, n - a, b, n - b]:
                        j = (i + d) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    return None, None

def _circulant_girth(n, a, b):
    """Compute the girth of C(n, {a, -a, b, -b}) using BFS."""
    adj = [[] for _ in range(n)]
    for d in [a, n - a, b, n - b]:
        for i in range(n):
            j = (i + d) % n
            adj[i].append(j)
    
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= 4: # Stop at depth 4 for girth 10
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth < 10:
                        return min_girth
        if min_girth < 10:
            return min_girth
    
    return min_girth if min_girth != float('inf') else float('inf')
