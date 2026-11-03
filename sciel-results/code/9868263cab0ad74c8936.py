import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        for n in range(242, 401):
            if time.time() > deadline:
                break
            
            max_tries = 50 if n < 260 else 10
            for _ in range(max_tries):
                if time.time() > deadline:
                    break
                edges = _generate_random_4_reg(n)
                if edges is None:
                    continue
                if _check_girth_fast(n, edges, g):
                    return n, edges

        return _luw_construct(k, g)
    return _luw_construct(k, g)

def _generate_random_4_reg(n):
    if n < 4: return None
    for _ in range(100):
        stubs = [i for i in range(n) for _ in range(4)]
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
        if valid:
            return list(edges)
    return None

def _check_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Verify degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
            
    r = (g - 1) // 2
    # For g=10, r=4.
    # We check if any cycle of length < g exists.
    # A cycle of length L < g implies there is an edge (u,v) such that
    # in BFS from some s, dist[u] + dist[v] + 1 < g.
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= r:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # v is visited.
                    # If v is parent of u, skip (tree edge).
                    if v == parent[u]:
                        continue
                    # Cycle length = dist[u] + dist[v] + 1
                    # Note: This assumes the shortest paths from s to u and s to v
                    # are disjoint except for s?
                    # No, this formula gives the length of the cycle formed by
                    # the two BFS paths and the edge (u,v).
                    # This is a valid cycle.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _luw_construct(k, g):
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
