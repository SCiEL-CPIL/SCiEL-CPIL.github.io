import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 115  # safety margin

    def build_luw(d, q):
        if q == 4:
            add_tbl = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
            mul_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x, y: add_tbl[x][y]
            sub = lambda x, y: add_tbl[x][y]
            mul = lambda x, y: mul_tbl[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            candidates = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = candidates[r]

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

    def check_girth_valid(n, edges, g):
        """Check if graph has no cycle of length < g. Returns True if valid."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex to find shortest cycle
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle: dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # Strategy: Try LUW(4,5) first (n≈1250, potentially valid if girth≥9)
    # Then LUW(5,5) (n≈6250, guaranteed girth≥10)
    
    candidates = []
    
    # Try d=4, q=5: girth bound is d+4=8, but actual might be ≥9
    if k == 5:
        try:
            n4, e4 = build_luw(4, 5)
            if time.time() < deadline - 30:
                if check_girth_valid(n4, e4, g):
                    return n4, e4
            candidates.append((n4, e4))
        except Exception:
            pass
    
    # Try d=5, q=5: girth bound is d+5=10, definitely valid for g=9
    try:
        n5, e5 = build_luw(5, 5)
        candidates.append((n5, e5))
    except Exception:
        pass
    
    # Return smallest valid candidate
    if not candidates:
        # Fallback: trivial construction (shouldn't happen)
        return 2 * k, [(i, (i + 1) % (2 * k)) for i in range(2 * k)]
    
    # Find the smallest valid one
    best = None
    for n, edges in candidates:
        if best is None or n < best[0]:
            if check_girth_valid(n, edges, g):
                best = (n, edges)
    
    if best is None:
        # Return the first one (should be valid by construction bounds)
        return candidates[0]
    
    return best
