import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
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
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    best_n, best_edges = luw_construct()
    
    # Try to find a smaller graph
    moore_bound = 2 * sum((k-1)**i for i in range(g//2))
    
    def get_girth(n, adj):
        max_depth = g // 2 - 1
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != start and dist[v] < dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth <= max_depth * 2 + 1:
                                return min_girth
        return min_girth if min_girth != float('inf') else 2 * max_depth + 2

    def random_k_regular(n, k):
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
            edges.add((min(u,v), max(u,v)))
        if not valid:
            return None
        return adj, list(edges)

    # Try a few random restarts
    for n in range(moore_bound, min(moore_bound + 20, best_n)):
        if time.time() > deadline:
            break
        if n % 2 != 0:
            continue
        # Try a few times for each n
        for _ in range(5):
            if time.time() > deadline:
                break
            adj, edges = random_k_regular(n, k)
            if not adj:
                continue
            girth = get_girth(n, adj)
            if girth >= g:
                best_n, best_edges = n, edges
                break
        if best_n == n:
            break

    return best_n, best_edges
