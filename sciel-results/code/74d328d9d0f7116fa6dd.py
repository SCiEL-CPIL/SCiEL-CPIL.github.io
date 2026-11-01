import time
import random
import sys
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def luw_construct():
        q = k
        d_candidates = []
        for d in range(3, 20):
            if d % 2 == 1:
                if d + 5 >= g:
                    d_candidates.append(d)
            else:
                if d + 4 >= g:
                    d_candidates.append(d)
        if not d_candidates:
            d = max(3, g - 5)
            if d % 2 == 0:
                d += 1
        else:
            d = min(d_candidates)
            
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = rules_cycle[r]
            
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
                yield (1 - side), tuple(y)
                
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def check_girth(n, edges, k, g, deadline):
        """Check if graph has girth at least g using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = (g - 1) // 2
        for start_v in range(n):
            if time.time() > deadline:
                return False
            visited = [[-1] * n for _ in range(max_depth + 1)]
            visited[0][start_v] = -2
            queue = deque([(start_v, 0)])
            while queue:
                v, depth = queue.popleft()
                if depth == max_depth:
                    continue
                for w in adj[v]:
                    if visited[depth + 1][w] == -1:
                        visited[depth + 1][w] = v
                        queue.append((w, depth + 1))
                    elif w != v and visited[depth + 1][w] != -2:
                        if depth + 1 < g:
                            return False
        return True

    def check_degree(n, edges, k):
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        return all(d == k for d in degree)

    def is_simple(n, edges):
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        return True

    def try_circulant(n, steps, k, g, deadline):
        """Try circulant graph C(n; steps) and check if it works."""
        edges = []
        for i in range(n):
            for s in steps:
                j = (i + s) % n
                if j > i:  # avoid duplicate edges
                    edges.append((i, j))
        
        if not check_degree(n, edges, k):
            return None
        if not is_simple(n, edges):
            return None
        if check_girth(n, edges, k, g, deadline):
            return edges
        return None

    # For k=4, g=10, try to find a smaller graph
    if k == 4 and g == 10:
        # Moore bound: 1 + 4*(1+3+9+27) = 161
        # Try circulant graphs with n from 162 to 320
        # Steps must be symmetric: {±a, ±b}
        # Try various (a, b) pairs
        
        # First, try known good constructions
        # The (4,10)-cage on 320 vertices can be constructed as a circulant
        # Try C(320; {±1, ±?}) - need to find appropriate steps
        
        best_n = float('inf')
        best_edges = None
        
        # Try circulant graphs
        # For 4-regular, steps = [a, -a, b, -b] with a != b, a != n-b, etc.
        # To avoid small cycles, a and b should be "independent"
        
        # Try n=320 with various step pairs
        for n in range(162, 321):
            if time.time() > deadline:
                break
            # Try a few step combinations
            # a should be small, b should be larger
            for a in range(1, n // 2 + 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n // 2 + 1):
                    if b == n - a or b == n - b:
                        continue
                    # Check that steps are distinct mod n
                    steps = [a, n - a, b, n - b]
                    if len(set(steps)) != 4:
                        continue
                    edges = try_circulant(n, steps, k, g, deadline)
                    if edges is not None:
                        if n < best_n:
                            best_n = n
                            best_edges = edges
                        break
                if best_edges is not None and n <= 200:
                    break
            if best_edges is not None and n <= 200:
                break
        
        if best_edges is not None:
            return best_n, best_edges
    
    # Fallback to LUW
    return luw_construct()
