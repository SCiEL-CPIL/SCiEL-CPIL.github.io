import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    if k != 4:
        return luw_construct(k, g)

    # Correct and fast girth check for circulant graphs
    def has_short_cycle(n, a, b, g_target):
        # Check if there is any cycle of length < g_target
        # For each vertex u, run BFS.
        # For each edge (x,y), if dist[u][x] + dist[u][y] + 1 < g_target, return True.
        
        # Precompute neighbors for speed?
        # Neighbors of i: (i+a)%n, (i-a)%n, (i+b)%n, (i-b)%n
        
        for u in range(n):
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                x = q.popleft()
                dx = dist[x]
                # Optimization: if dx >= g_target // 2, we might still need to check edges?
                # No, if dx >= g/2, then dist[x] + dist[y] + 1 >= g/2 + 0 + 1.
                # If g=10, g/2=5. 5+0+1=6 < 10.
                # So we need to check all nodes.
                
                # Check edges from x
                for y in ((x + a) % n, (x - a) % n, (x + b) % n, (x - b) % n):
                    if dist[y] == -1:
                        dist[y] = dx + 1
                        q.append(y)
                    else:
                        # y is already visited
                        # Check cycle length
                        if dist[x] + dist[y] + 1 < g_target:
                            return True
        
        return False

    # Search for circulant graph C_n(a, b)
    # We want to minimize n.
    # Lower bound for (4,10) is 243.
    # We will search from 243 upwards.
    
    for n in range(243, 321):
        if time.time() > deadline:
            break
            
        # Symmetry: a < b, a < n/2, b < n/2
        # Also a != b, a != n-b
        # We can restrict a to 1..n//2, b to a+1..n//2
        # But we must ensure steps are distinct.
        
        found = False
        for a in range(1, n // 2 + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, n // 2 + 1):
                # Check distinctness
                if a == b or a == n - b:
                    continue
                # Also check if 2a == 0 or 2b == 0 (loops/multi-edges)
                # a != n-a => 2a != n
                # b != n-b => 2b != n
                if 2 * a == n or 2 * b == n:
                    continue
                if a == 0 or b == 0:
                    continue
                
                # Check girth
                if not has_short_cycle(n, a, b, g):
                    # Valid!
                    edges = []
                    for i in range(n):
                        for s in (a, n - a, b, n - b):
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return luw_construct(k, g)
