import time
import random
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Fallback: LUW graph
    def luw_construct():
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
        i = 0
        while i < len(order):
            if time.time() > deadline:
                break
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        n = len(order)
        return n, edges

    # Try Circulant Search
    # k=5, g=9
    if k == 5 and g == 9:
        def girth_ok_fast(n, steps):
            """Check if circulant graph has girth >= g using BFS.
            steps is a list of offsets.
            """
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for i in range(n):
                for s in steps:
                    j = (i + s) % n
                    if i < j:
                        adj[i].append(j)
                        adj[j].append(i)
            
            # Check girth
            # Girth is the length of the shortest cycle.
            # For each vertex, BFS up to depth g//2.
            # If we find a back edge to a vertex at depth < g//2, cycle is short.
            # Actually, BFS from each vertex to find shortest cycle is O(n * E).
            # For n~1000, E~2500, this is fast.
            
            # Optimization: Only need to check from vertices 0 to n/2?
            # No, just do it for all, but break early.
            
            for start in range(n):
                # BFS
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                
                while q:
                    u = q.popleft()
                    if dist[u] >= g // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            # Found a cycle
                            # Length = dist[u] + dist[v] + 1
                            # But we need to be careful with the root.
                            # If v is an ancestor of u, we have a cycle.
                            # Since we are doing BFS, the first time we see a visited node (not parent),
                            # the cycle length is dist[u] + dist[v] + 1.
                            # However, this only finds cycles where the edge (u,v) closes the loop.
                            # This is correct for girth.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
            return True

        def make_circulant_edges(n, s1, s2):
            h = n // 2
            steps = (s1, -s1, s2, -s2, h)
            edges = []
            for i in range(n):
                for s in steps:
                    j = (i + s) % n
                    if i < j:
                        edges.append((i, j))
            return edges

        best_result = None
        best_n = float('inf')
        
        # Search for circulant graphs
        # Moore bound for k=5, g=9: 426
        # n must be even
        n_start = 428
        # Search up to 1500 to beat LUW (1562)
        n_end = 1500
        
        for n in range(n_start, n_end + 1, 2):
            if time.time() > deadline - 5:
                break
            h = n // 2
            # s1, s2 candidates
            # To reduce search, try small s1, s2
            # s1, s2 in [1, h-1]
            # Limit to first 50 candidates for speed
            s1_candidates = list(range(1, min(50, h)))
            s2_candidates = list(range(1, min(50, h)))
            
            for s1 in s1_candidates:
                if time.time() > deadline - 5:
                    break
                for s2 in s2_candidates:
                    if s1 == s2:
                        continue
                    if s1 == h or s2 == h:
                        continue
                    # Check girth
                    if girth_ok_fast(n, (s1, -s1, s2, -s2, h)):
                        if n < best_n:
                            best_n = n
                            best_result = (n, make_circulant_edges(n, s1, s2))
                        break
                if best_result and best_n <= n:
                    break
            if best_result and best_n <= n:
                break
        
        if best_result:
            return best_result
        
        # Fallback to LUW
        return luw_construct()
        
    return luw_construct()
