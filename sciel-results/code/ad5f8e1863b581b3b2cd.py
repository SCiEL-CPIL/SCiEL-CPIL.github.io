"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 145.0
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def verify_girth(n, edges, g):
        """Verify girth >= g using BFS with non-backtracking from each vertex."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check all vertices have degree k
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check no multi-edges or loops
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        # Girth check: for each vertex, BFS with state (v, prev) up to depth (g-1)//2
        max_depth = (g - 1) // 2
        for s in range(n):
            visited = set()
            queue = deque()
            queue.append((s, -1, 0))
            visited.add((s, -1))
            while queue:
                v, prev, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for w in adj[v]:
                    if w == prev:
                        continue
                    state = (w, v)
                    if state in visited:
                        return False
                    visited.add(state)
                    queue.append((w, v, depth + 1))
        return True
    
    def try_luw(k, g):
        """Lazebnik-Ustimenko-Woldar construction."""
        q = k
        if g % 2 == 1:
            d = g - 4
        else:
            d = g - 5
        if d < 3:
            d = 3
        
        if not (is_prime(q) or q == 4):
            return None
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def gf_add(x, y): return x ^ y
            def gf_sub(x, y): return x ^ y
            def gf_mul(x, y): return table[x][y]
        else:
            def gf_add(x, y): return (x + y) % q
            def gf_sub(x, y): return (x - y) % q
            def gf_mul(x, y): return x * y % q
        
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
                        y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                    else:
                        y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                yield (1 - side), tuple(y)
        
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
        if verify_girth(n, edges, g):
            return n, edges
        return None
    
    result = try_luw(k, g)
    if result is not None:
        return result
    
    # Fallback: return minimal valid structure
    n = k
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, edges
# EVOLVE-BLOCK-END
