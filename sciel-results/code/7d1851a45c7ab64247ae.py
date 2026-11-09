import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # 1. Try heuristic search for a smaller graph
    # Target n: known best is 1116. Try up to 2000.
    # If we find a valid graph, return it.
    
    def is_valid_graph(n, edges):
        # Check simple, k-regular, girth >= g
        if len(edges) != n * k // 2:
            return False
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (min(u, v), max(u, v)) in edge_set:
                return False
            edge_set.add((min(u, v), max(u, v)))
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth
        for start in range(n):
            # BFS to find shortest cycle involving start
            # We need to ensure no cycle shorter than g
            # Standard BFS from start doesn't find all cycles.
            # We need to check for cycles of length < g.
            # A cycle of length L < g implies there is a path of length < g/2 from some node?
            # Actually, we can just do BFS from each node and check for back-edges?
            # No, BFS tree doesn't capture all cycles.
            # However, if girth >= g, then the graph is a tree up to distance (g-1)/2.
            # So we can check if the ball of radius r = (g-1)//2 around each vertex is a tree.
            r = (g - 1) // 2
            for start in range(n):
                # BFS to depth r
                visited = {start: 0}
                q = deque([(start, 0)])
                while q:
                    u, dist = q.popleft()
                    if dist == r:
                        continue
                    for v in adj[u]:
                        if v not in visited:
                            visited[v] = dist + 1
                            q.append((v, dist + 1))
                        elif v != start and dist + 1 < r: # Found a cycle?
                            # If v is visited and v != parent, and dist+1 < r, then cycle length < 2*r+1 = g
                            # We need to be careful with parent check.
                            pass # This logic is flawed.
                
                # Better: Just check if the number of vertices in ball of radius r is 1 + k * sum_{i=0}^{r-1} (k-1)^i
                # If it is less, there is a cycle.
                # If it is more, impossible.
                # Actually, if girth >= g, the ball of radius r is a tree.
                # So the number of vertices in the ball must be exactly 1 + k * ((k-1)^r - 1) / (k-2)
                expected_count = 1 + k * ((k-1)**r - 1) // (k-2) if k > 2 else 1 + k*r
                if len(visited) < expected_count:
                    return False
        return True

    def generate_random_regular(n, k):
        # Generate random k-regular graph
        # Use configuration model
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        random.shuffle(half_edges)
        edges = []
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u != v:
                edges.append((u, v))
        # Remove multi-edges and loops
        # This is tricky. Just retry if invalid.
        return edges

    # Heuristic search
    for n in range(1116, 2001, 50):
        if time.time() > deadline:
            break
        for _ in range(100): # Try 100 random graphs
            if time.time() > deadline:
                break
            edges = generate_random_regular(n, k)
            # Filter loops/multi-edges
            # This is expensive. Skip for now, assume simple.
            if is_valid_graph(n, edges):
                return n, edges
    
    # 2. Fallback to LUW construction
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
