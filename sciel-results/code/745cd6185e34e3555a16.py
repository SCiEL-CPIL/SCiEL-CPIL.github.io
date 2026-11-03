import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 20s margin before 120s kill

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

    def check_girth_bfs(n, adj, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Back edge or cross edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def find_circulant(k, g, max_n=None):
        """Search for a circulant graph with k generators and girth >= g."""
        if max_n is None:
            # Moore bound
            r = (g - 1) // 2
            moore = 1
            for i in range(r):
                moore += k * (k - 1) ** i
            max_n = moore + 200

        # For k=5, g=9: Moore bound is 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1 + 5 + 20 + 80 + 320 = 426
        # Try n from Moore bound to max_n
        for n in range(428, max_n + 1, 2):  # n must be even for symmetric generators
            if time.time() > deadline:
                break
            # Try random sets of k generators
            # For k=5, we need 5 generators, but since graph is undirected, 
            # generators come in pairs (a, -a) or we use a single generator with both +a and -a
            # Actually for k=5, we need 5 distinct neighbors per vertex
            # In a circulant, vertex i connects to i+g1, i-g1, i+g2, i-g2, i+g3, i-g3, ...
            # So for k=5, we need generators such that the set {±g1, ±g2, ±g3, ..., ±g5} has size 5
            # This means some generators must be self-inverse (g = -g mod n), i.e., 2g = 0 mod n
            # Or we use a different structure
            
            # For simplicity, let's try k=3 first with proper circulant search
            if k == 3:
                for _ in range(500):
                    if time.time() > deadline:
                        break
                    gen = random.sample(range(1, n // 2 + 1), 3)
                    adj = [[] for _ in range(n)]
                    valid = True
                    for u in range(n):
                        for v in gen:
                            adj[u].append((u + v) % n)
                            adj[u].append((u - v) % n)
                        adj[u] = list(set(adj[u]))
                        if len(adj[u]) != k:
                            valid = False
                            break
                    if not valid:
                        continue
                    if check_girth_bfs(n, adj, g):
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        return n, edges
        return None

    # Get LUW construction as baseline
    best_n, best_edges = luw_construct(k, g)

    # Try circulant search for k=3
    if k == 3 and time.time() < deadline - 5:
        circ = find_circulant(k, g)
        if circ:
            n_c, e_c = circ
            if n_c < best_n:
                best_n, best_edges = n_c, e_c

    return best_n, best_edges
