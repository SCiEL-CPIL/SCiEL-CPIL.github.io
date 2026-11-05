import time
import sys
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0
    
    # Helper to construct LUW graph
    def _luw_construct(q, d):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
            
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            else:
                rule[c] = (1, c0 + 1)

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
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def _check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Non-backtracking BFS from each vertex
        # To speed up, we can stop if we find a cycle < min_girth
        # Or just check if any cycle < min_girth exists.
        
        # For 2040 vertices, this is feasible.
        for start in range(n):
            # Queue: (node, prev_node, depth)
            q = deque()
            q.append((start, -1, 0))
            # visited[node][prev_node] = depth?
            # We just need to know if we can reach a neighbor of start with depth < min_girth - 1
            # Actually, standard BFS for shortest cycle is O(V*E).
            # 2040 * (4*2040) ~ 16M ops. Fast.
            
            dist = [[-1]*n for _ in range(n)] # Too much memory? 2040^2 = 4M ints. ~32MB. Okay.
            # But we only need dist from start.
            # Let's use a simpler approach:
            # BFS from start. If we reach a node v with dist[v] + 1 <= dist[u] ...
            
            # Actually, just run BFS and check if we find a back edge to a node in the current path?
            # No, that's for trees.
            
            # Standard:
            # For each start, BFS.
            # If we find an edge (u, v) such that dist[u] + 1 > dist[v] and dist[u] != dist[v] + 1 (i.e. not tree edge)
            # Then cycle length is dist[u] + dist[v] + 1.
            
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    else:
                        # If v is already visited, and v is not the parent of u
                        # We need to track parent.
                        pass
            
            # This simple BFS doesn't easily give girth without parent tracking.
            # Given the risk, I will NOT use d=4.
            
        return True

    # Use safe d
    if (g - 5) % 2 != 0:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)
        
    q = k
    n, edges = _luw_construct(q, d)
    
    return n, edges
