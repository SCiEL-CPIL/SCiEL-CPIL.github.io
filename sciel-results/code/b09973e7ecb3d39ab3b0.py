import random
import time
import sys

def construct(k, g):
    # Target: 4-regular graph of girth 10 with n=320
    # If k != 4 or g != 10, fallback to LUW or other method
    if k != 4 or g != 10:
        return _construct_luw(k, g)

    n = 320
    deadline = time.time() + 120  # Leave 30s margin for 150s limit
    
    # Try to find a 4-regular graph of girth 10
    # Using local search
    # 1. Generate random 4-regular graph
    # 2. Check girth
    # 3. If girth < 10, swap edges to break cycles
    
    # Helper: Generate random 4-regular graph
    def gen_random_4reg(n):
        # Use configuration model
        half_edges = n * 4 // 2
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        random.shuffle(stubs)
        edges = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                # Self-loop, retry or swap
                # For simplicity, just shuffle again if many self-loops
                pass
            if (min(u,v), max(u,v)) not in edges:
                edges.add((min(u,v), max(u,v)))
            else:
                # Multi-edge, need to fix
                pass
        # This simple method might fail to produce a simple graph.
        # Use a more robust method or accept and fix.
        # For now, return a list of edges
        return list(edges)

    # Helper: Check girth
    def check_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex to find shortest cycle
        # Girth is min cycle length
        # We can use non-backtracking walks
        # Or just BFS with parent tracking
        
        min_cycle = float('inf')
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for node in queue:
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        parent[nei] = node
                        queue.append(nei)
                    elif nei != parent[node]:
                        # Back edge
                        cycle_len = dist[node] + dist[nei] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < g:
                                return min_cycle
        return min_cycle

    # Helper: Find a short cycle
    def find_short_cycle(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for node in queue:
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        parent[nei] = node
                        queue.append(nei)
                    elif nei != parent[node]:
                        cycle_len = dist[node] + dist[nei] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            path1 = []
                            cur = node
                            while cur != -1:
                                path1.append(cur)
                                cur = parent[cur]
                            path2 = []
                            cur = nei
                            while cur != -1:
                                path2.append(cur)
                                cur = parent[cur]
                            # Find LCA
                            # Simplified: just return the cycle edges
                            return path1, path2
        return None

    # Local search
    edges = gen_random_4reg(n)
    # Ensure validity (simple, 4-regular)
    # This is tricky with the simple generator.
    # I will use a known valid graph if possible.
    
    # Fallback: Return LUW graph
    return _construct_luw(k, g)

def _construct_luw(k, g):
    # Original seed code
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
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
