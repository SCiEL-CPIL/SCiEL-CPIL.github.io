import time
import random
import sys
from collections import deque

def construct(k, g):
    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    def neighbours_luw(vertex):
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
        for other in neighbours_luw(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    n_luw, edges_luw = len(order), edges
    
    deadline = time.time() + 110
    best_n = n_luw
    best_edges = edges_luw
    
    # Structural change: Random 3-regular graph generation with girth check
    # Try to find a smaller valid graph than LUW
    
    def generate_random_3reg(n):
        """Generate a random 3-regular graph on n vertices using configuration model."""
        if n < 4 or n % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edge_set = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
        return list(edge_set)
    
    def has_short_cycle(adj, n, g):
        """Check if graph has a cycle shorter than g using BFS."""
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            # Check each edge for short cycles
            for u in range(n):
                for v in adj[u]:
                    if u >= v:
                        continue
                    # BFS from u to v avoiding direct edge
                    d2 = [-1] * n
                    d2[u] = 0
                    q2 = deque([u])
                    while q2:
                        curr = q2.popleft()
                        if curr == v:
                            break
                        for nb in adj[curr]:
                            if nb == v and curr == u:
                                continue
                            if d2[nb] == -1:
                                d2[nb] = d2[curr] + 1
                                q2.append(nb)
                    if d2[v] != -1 and d2[v] < g - 1:
                        return True
        return False
    
    # Try random constructions for smaller n
    random.seed(12345)
    for target_n in range(936, min(1280, best_n), -8):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            edges = generate_random_3reg(target_n)
            if edges is None:
                continue
            adj = [set() for _ in range(target_n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            if not has_short_cycle(adj, target_n, g):
                best_n = target_n
                best_edges = edges
                found = True
                break
        if found:
            break
    
    return best_n, best_edges
