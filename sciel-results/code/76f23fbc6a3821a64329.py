import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_graph():
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

    # Helper: Generate random k-regular graph
    def random_regular_graph(n, k):
        if n * k % 2 != 0:
            return None
        half_edges = []
        for i in range(n):
            half_edges.extend([i] * k)
        
        while True:
            random.shuffle(half_edges)
            edges = []
            valid = True
            seen = set()
            for i in range(0, n * k, 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in seen:
                    valid = False
                    break
                seen.add((u, v))
                edges.append((u, v))
            if valid:
                return edges

    # Helper: Check girth
    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Main Logic
    # If k=5 and g=9, try to beat the baseline (1250)
    if k == 5 and g == 9:
        # Try N values. 1116 is known to exist.
        # Start with 1116, then 1200, then 1000.
        target_ns = [1116, 1150, 1200, 1000, 900]
        
        for n in target_ns:
            if time.time() > deadline - 10:
                break
                
            # Try a few random instances
            max_attempts = 50
            for _ in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                edges = random_regular_graph(n, k)
                if edges is not None:
                    if check_girth(n, edges, g):
                        return n, edges
                        
    # Fallback to LUW
    return luw_graph()
