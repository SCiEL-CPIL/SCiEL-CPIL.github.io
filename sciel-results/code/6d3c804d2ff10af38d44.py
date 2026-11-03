import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Helper to build LUW graph CD(d, q)
    def build_luw(d, q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length < g
        # Since graph is k-regular and likely bipartite (LUW is bipartite),
        # we check for even cycles of length 4, 6, ..., g-2
        # BFS from each vertex up to depth g//2
        # If we find a back-edge to a node in the current path, we have a cycle.
        # Simpler: BFS from each node, track parent. If we visit a node already visited
        # in the same BFS tree (and it's not the parent), we have a cycle.
        # Wait, standard BFS for girth:
        # For each start node, do BFS. If we find a node that was already visited 
        # in this BFS (and is not the parent), the cycle length is dist[u] + dist[v] + 1.
        # We need to ensure no such cycle has length < g.
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q_bfs = deque()
            q_bfs.append(s)
            dist[s] = 0
            while q_bfs:
                u = q_bfs.popleft()
                if dist[u] >= g // 2: 
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q_bfs.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # Determine minimum d
    # LUW girth is 2d+2. Need 2d+2 >= g => d >= (g-2)/2
    min_d = max(3, (g - 2) // 2)
    
    # Try d values starting from min_d
    # Cap d to avoid infinite loops or huge graphs if something goes wrong
    for d in range(min_d, min_d + 5):
        if time.time() > deadline:
            break
        n, edges = build_luw(d, k)
        # Quick check: is it k-regular?
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(x == k for x in deg):
            if check_girth(n, edges, g):
                return n, edges
                
    # Fallback to seed's conservative d if all above fail
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = build_luw(d_seed, k)
    return n, edges
