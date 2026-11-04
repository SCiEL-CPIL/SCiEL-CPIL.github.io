import time
import random
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before the 120s limit

    def check_girth_fast(n, adj, min_girth):
        """Check if the graph has girth at least min_girth using BFS from each vertex."""
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                du = dist[u]
                if du + 1 >= min_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
                    elif dist[v] <= du + 1:
                        # Found a cycle of length du + 1 + dist[v]
                        if du + 1 + dist[v] < min_girth:
                            return False
        return True

    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph construction."""
        q = k
        d = max(3, (g - 1) // 2)
        
        if q == 4:
            gf4_mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: gf4_mul_table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Standard LUW rule
        rule = {}
        for c in range(2, d + 1):
            rule[c] = (1, c - 1)

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
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges

    def generate_4_regular(n):
        """Generate a random 4-regular graph on n vertices."""
        if n < 4 or n % 2 != 0:
            return None
        
        for _ in range(50):
            stubs = []
            for i in range(n):
                for _ in range(4):
                    stubs.append(i)
            
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            
            if not valid:
                continue
            
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != 4 for d in deg):
                continue
            
            return list(edges)
        
        return None

    def edges_to_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    if k == 4 and g == 10:
        n_luw, edges_luw = luw_construct(k, g)
        best_n, best_edges = n_luw, edges_luw
        
        # Try to find a smaller graph
        moore_bound = 1 + k * sum((k-1)**i for i in range((g-2)//2 + 1))
        
        for n_try in range(max(moore_bound, 244), min(n_luw, 400)):
            if time.time() > deadline:
                break
            if n_try % 2 != 0:
                continue
            
            for attempt in range(10):
                if time.time() > deadline:
                    break
                edges = generate_4_regular(n_try)
                if edges is None:
                    continue
                adj = edges_to_adj(n_try, edges)
                if check_girth_fast(n_try, adj, g):
                    best_n, best_edges = n_try, edges
                    break
            if best_n == n_try:
                break
        
        return best_n, best_edges
    else:
        return luw_construct(k, g)
