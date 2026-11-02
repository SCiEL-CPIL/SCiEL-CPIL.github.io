import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Moore bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2 - 1
        moore = k * sum((k - 1) ** i for i in range(r))
        
    if moore < k + 1:
        moore = k + 1

    def check_girth(n, edges, target_g):
        """Check if graph has girth at least target_g using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        radius = (target_g - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    elif dist[v] != dist[u] + 1 and dist[v] < dist[u]:
                        # Back edge, check cycle length
                        if dist[u] + dist[v] + 1 < target_g:
                            return False
        return True

    def luw_construct(k, g):
        """LUW construction for k-regular graphs of given girth."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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

    # For k=5, g=9, try systematic circulant search
    if k == 5 and g == 9:
        # Known best is 1116, try to find something better or at least valid
        # Try circulant C_n(a, b, n/2) for even n
        best_n = None
        best_edges = None
        
        # Search range: from moore bound up to 1200
        n_start = max(moore, 100)
        n_end = min(1200, n_start + 500)
        
        for n in range(n_start, n_end + 1, 2):  # n must be even
            if time.time() > deadline - 10:
                break
            
            # Try systematic pairs (a, b)
            for a in range(1, n // 2):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, n // 2):
                    if a + b == n:
                        continue
                    
                    # Quick check: verify no short cycles
                    # Build adjacency and check girth
                    adj = [[] for _ in range(n)]
                    for i in range(n):
                        for delta in (a, -a, b, -b, n // 2):
                            j = (i + delta) % n
                            adj[i].append(j)
                    
                    # Check if simple (no multi-edges)
                    edge_set = set()
                    valid = True
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                if (i, j) in edge_set:
                                    valid = False
                                    break
                                edge_set.add((i, j))
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Check girth using BFS
                    radius = (g - 1) // 2
                    girth_ok = True
                    for start in range(n):
                        dist = [-1] * n
                        dist[start] = 0
                        q = deque([start])
                        while q:
                            u = q.popleft()
                            if dist[u] >= radius:
                                continue
                            for v in adj[u]:
                                if dist[v] == -1:
                                    dist[v] = dist[u] + 1
                                    q.append(v)
                                elif dist[v] < dist[u] and dist[u] + dist[v] + 1 < g:
                                    girth_ok = False
                                    break
                            if not girth_ok:
                                break
                        if not girth_ok:
                            break
                    
                    if girth_ok:
                        edges = []
                        for i in range(n):
                            for delta in (a, b, n // 2):
                                j = (i + delta) % n
                                if i < j:
                                    edges.append((i, j))
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                        break  # Found valid for this n, move to next n
        
        if best_n is not None:
            return best_n, best_edges
    
    # Fallback: LUW construction
    return luw_construct(k, g)
