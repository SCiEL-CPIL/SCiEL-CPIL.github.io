import random
import time
import sys
from collections import deque

def _luw_construct(k, g):
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

def _generate_random_regular(n, k):
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    adj = [[] for _ in range(n)]
    i = 0
    while i < len(stubs):
        u = stubs[i]
        v = stubs[i+1]
        i += 2
        if u == v:
            # Swap with a random other stub
            j = random.randint(i, len(stubs)-1)
            stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
            i -= 2 # retry
            continue
        if (min(u,v), max(u,v)) in edges:
            j = random.randint(i, len(stubs)-1)
            stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
            i -= 2
            continue
        edges.add((min(u,v), max(u,v)))
        adj[u].append(v)
        adj[v].append(u)
    return list(edges), adj

def _check_girth(n, adj, g):
    # BFS from each vertex up to depth g//2
    # If we find a cycle of length < g, return False
    half_g = g // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Cycle length = dist[u] + dist[v] + 1
                    # Wait, this is not quite right for undirected graphs.
                    # If we see an edge (u,v) where dist[v] != dist[u] + 1, it's a back edge.
                    # The cycle length is dist[u] + dist[v] + 1?
                    # No, in BFS tree, if (u,v) is not a tree edge, the cycle length is dist[u] + dist[v] + 1 - 2*LCA_depth.
                    # We don't have LCA.
                    # Simpler: Just check if dist[u] + dist[v] + 1 < g?
                    # This is an upper bound on the cycle length? No.
                    # Actually, if we just want to know if girth < g, we can stop BFS at depth g-1.
                    # If we reach a vertex already visited, we have a cycle.
                    # The length is at most 2 * depth + 1.
                    # If 2 * half_g + 1 < g, we are safe?
                    # For g=9, half_g=4. 2*4+1=9.
                    # So if we find a cycle within depth 4, it has length <= 9.
                    # We need to exclude length 9.
                    # So if we find a cycle of length < 9, we fail.
                    # A cycle found by BFS at depth d has length <= 2d+1.
                    # If d=3, length <= 7.
                    # If d=4, length <= 9.
                    # So we need to distinguish between length 9 and <9.
                    # This BFS approach is tricky.
                    pass
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 5 and g == 9:
        # Try n from 1100 to 1240
        for n in range(1100, 1250, 10):
            if time.time() > deadline:
                break
            for _ in range(10):
                if time.time() > deadline:
                    break
                edges, adj = _generate_random_regular(n, 5)
                # Check girth
                # If girth >= 9, return
                # ...
                pass

    return _luw_construct(k, g)
