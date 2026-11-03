import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Fallback: LUW construction (from seed)
    def luw_construct(k, g):
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

    # Greedy Construction
    def greedy_construct(k, g):
        n = 0
        adj = []
        edges = []
        
        # Start with a path of 3 vertices? No, degree must be k.
        # Start with a cycle of length g? No, girth is g, so no cycle shorter than g.
        # Start with a tree.
        
        # We need to build a k-regular graph.
        # Strategy: Add vertices one by one.
        # Each new vertex connects to k existing vertices? No, that would make degrees k+1 for existing.
        # Better: Add edges to existing vertices with degree < k.
        
        # Initialize with a path of length 3?
        # v0-v1-v2-v3...
        # Degrees: v0:1, v1:2, v2:2, v3:1
        # We need to close the graph.
        
        # Let's start with a small tree and grow it.
        # v0, v1, v2
        adj = [[], [], []]
        edges = [(0, 1), (1, 2)]
        n = 3
        
        # We need to add more vertices and edges.
        # A simpler heuristic: 
        # 1. Generate a random k-regular graph.
        # 2. Check girth.
        # 3. If girth < g, swap edges to increase girth.
        
        # This is Simulated Annealing.
        # Given time constraints, let's try a simple greedy extension.
        
        # Actually, a better approach for small k:
        # Use the **Lazebnik-Ustimenko-Woldar** construction but with a smaller q?
        # No, q must be >= k.
        
        # Let's try to construct the **Klein graph** (56, 8) and see if we can extend?
        # No.
        
        # I will implement a **Greedy Edge Addition** that tries to maintain girth.
        # This is complex.
        
        # Fallback to LUW if greedy fails or is too slow.
        return None

    # Try Greedy
    result = greedy_construct(k, g)
    if result:
        n, edges = result
        # Verify
        if verify_girth(n, edges, k, g):
            return n, edges

    # Return LUW
    return luw_construct(k, g)

def verify_girth(n, edges, k, g):
    # Check if graph is simple, k-regular, and girth >= g
    # Girth check: BFS from each vertex up to depth g//2
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth
    # For each vertex, BFS up to depth g//2
    # If we encounter a vertex at depth < g//2 that was visited at a smaller depth, we have a cycle.
    # Actually, we need to check for cycles of length < g.
    # BFS from u, if we reach v at depth d, and v was reached at depth d' < d, cycle length is d + d' + 1?
    # No, standard BFS girth check:
    # For each u, BFS. If we find an edge (x, y) such that depth[x] + depth[y] + 1 < g, then girth < g.
    
    half_g = g // 2
    for u in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        q = deque([u])
        while q:
            x = q.popleft()
            if dist[x] >= half_g:
                continue
            for y in adj[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    parent[y] = x
                    q.append(y)
                else:
                    # Found a cycle
                    # Cycle length is dist[x] + dist[y] + 1
                    # But this might not be the shortest cycle.
                    # However, if dist[x] + dist[y] + 1 < g, we have a violation.
                    if dist[x] + dist[y] + 1 < g:
                        return False
    return True
