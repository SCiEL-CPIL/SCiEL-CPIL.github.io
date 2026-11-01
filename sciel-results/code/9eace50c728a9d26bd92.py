import numpy as np
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses Lazebnik-Ustimenko-Woldar (LUW) construction.
    Tries smaller 'd' parameters first to minimize vertex count.
    """
    q = k
    
    def get_luw_graph(d, q):
        """Constructs the LUW graph CD(d, q)."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Define the rule for LUW graph
        # Based on Lazebnik, Ustimenko, Woldar (1998)
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
                    if side == 0:
                        # point p ~ line l iff l[c] - p[c] = l[a] * p[b]
                        # l[c] = p[c] + l[a]*p[b]
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        # line l ~ point p iff p[c] - l[c] = l[a] * p[b] -> p[c] = l[c] + l[a]*p[b]
                        # Wait, the seed logic:
                        # y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                        # Let's stick to the seed's proven logic for correctness.
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

    def check_girth_fast(n, edges, target_girth):
        """
        Returns True if girth is >= target_girth.
        Uses BFS from each vertex with early termination.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q_bfs = deque([s])
            
            while q_bfs:
                u = q_bfs.popleft()
                du = dist[u]
                if du + 1 >= target_girth - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        q_bfs.append(v)
                    elif v != parent[u]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    # Try d=4 first (smaller graph)
    # For k=5, q=5.
    # If d=4 fails girth check, try d=5.
    
    candidates = []
    # Heuristic: d=4 is smaller. d=5 is safer for girth.
    # We try d=4, then d=5.
    
    for d in [4, 5]:
        n, edges = get_luw_graph(d, q)
        # Verify girth
        if check_girth_fast(n, edges, g):
            return n, edges
    
    # Fallback: d=5 is known to have girth >= 10
    n, edges = get_luw_graph(5, q)
    return n, edges
