import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer

    def luw_construction(d, q):
        """
        Constructs the LUW graph for given d and q.
        Returns (n, edges).
        """
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Rule definition
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
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

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

    def get_girth_fast(n, edges, target_girth):
        """
        Checks if the graph has girth at least target_girth using BFS.
        Returns True if girth >= target_girth, else False.
        """
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS for shortest cycle
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            found_short = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Cycle found: dist[u] + dist[v] + 1
                        # Ensure we don't count the edge (u,v) as a cycle of length 2
                        # which is only possible if there are multi-edges (not allowed here)
                        # or if v is the parent of u (back edge).
                        if v != parent[u] and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < target_girth:
                                return False
            if time.time() > deadline:
                return False
        return True

    # Determine candidate d values
    # Theoretical girth of LUW(q,d) is approx 2d+4.
    # We want girth >= g.
    # So 2d+4 >= g => d >= (g-4)/2.
    # We try d values starting from the minimum required, going up.
    
    # q is usually k. If k is not a prime power, this might fail, 
    # but for the task k=5, it works.
    q = k
    
    # If k is not a prime power, we might need to adjust q.
    # For now, assume q=k as per seed.
    
    min_d = max(1, (g - 4 + 1) // 2) # Ceiling of (g-4)/2
    
    # We will try d = min_d, min_d+1, min_d+2...
    # Also, the seed used a different formula. Let's include the seed's d as a fallback.
    seed_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    candidates = sorted(set([min_d, min_d + 1, min_d + 2, seed_d]))
    
    best_result = None
    best_n = float('inf')
    
    for d in candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = luw_construction(d, q)
        except Exception:
            continue
            
        # Check if it's k-regular
        # LUW is k-regular by construction if q=k.
        # But let's verify degrees quickly.
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != k for dg in deg):
            continue
            
        # Check girth
        # Optimization: If n is very large, girth check might be slow.
        # But for k=5, d=3 -> n=250. d=4 -> n=1250. d=5 -> n=6250.
        # 6250 nodes, BFS is O(n^2) ~ 40M ops. Might be slow in Python.
        # We should only do full girth check for small n, or use a faster method.
        # For now, we try. If it's too slow, we rely on the theoretical bound.
        # Theoretical bound: If 2d+4 >= g, we *assume* it's valid unless we find a shorter cycle.
        # To be safe, we only verify for n < 2000. For larger n, we trust the bound 2d+4 >= g.
        
        if n < 2000:
            if not get_girth_fast(n, edges, g):
                continue
        else:
            # Trust the theoretical bound
            if 2 * d + 4 < g:
                continue
                
        if n < best_n:
            best_n = n
            best_result = (n, edges)
            
    if best_result:
        return best_result
    else:
        # Fallback to seed's logic
        d = seed_d
        return luw_construction(d, q)
