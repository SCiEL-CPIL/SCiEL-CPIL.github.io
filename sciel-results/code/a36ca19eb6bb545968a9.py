import time
import random
from collections import deque

def _verify_girth_circulant(n, steps, g):
    """
    Verify girth of a circulant graph C(n, steps) by BFS from 0.
    """
    adj = []
    for u in range(n):
        neighbors = [(u + s) % n for s in steps]
        adj.append(neighbors)
        
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    q = deque([0])
    
    while q:
        u = q.popleft()
        if dist[u] >= g - 1:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
            elif v != parent[u]:
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g:
                    return False
    return True

def _construct_luw(k, g):
    """Fallback LUW construction."""
    q = k
    d_min = max(3, (g - 2 + 1) // 2)
    if d_min % 2 == 0:
        d_min += 1
    d = d_min
    
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
    n = len(order)
    return n, edges

def construct(k, g):
    deadline = time.time() + 115
    
    # Moore bound
    if g % 2 == 0:
        r = g // 2 - 1
        n_min = 2 * sum((k-1)**i for i in range(r+1))
    else:
        r = (g - 1) // 2
        n_min = 1 + k * sum((k-1)**i for i in range(r))
        
    if k == 5:
        # Fallback is LUW (n=1250 for g=9)
        best_n, best_edges = _construct_luw(k, g)
        
        # Search for circulant in range [n_min, 1116]
        # 1116 is the known best. If we find < 1116, we win.
        # If we find 1116, we tie with best known, but beat LUW.
        
        start_n = n_min if n_min % 2 == 0 else n_min + 1
        end_n = 1116
        
        for n in range(start_n, end_n + 1, 2):
            if time.time() > deadline - 5:
                break
            
            # Try random step pairs
            # We want to avoid 4-cycles: a+b != n, a-b != 0, 2a != n, 2b != n
            # We want to avoid 6-cycles etc.
            
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts:
                attempts += 1
                if time.time() > deadline - 2:
                    break
                
                candidates = list(range(1, n//2))
                if len(candidates) < 2:
                    continue
                
                a, b = random.sample(candidates, 2)
                
                # Quick filters
                if 2*a == n or 2*b == n:
                    continue
                if a + b == n:
                    continue
                if a == b: # sample prevents this
                    continue
                
                steps = [n//2, a, n-a, b, n-b]
                
                if len(set(steps)) != 5:
                    continue
                    
                if _verify_girth_circulant(n, steps, g):
                    edges = []
                    for u in range(n):
                        for s in steps:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                    
        return best_n, best_edges
    else:
        return _construct_luw(k, g)
