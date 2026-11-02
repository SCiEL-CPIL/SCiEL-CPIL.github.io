import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
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

    def nonbacktracking_girth(adj, g):
        n = len(adj)
        for s in range(n):
            for a in adj[s]:
                dist = {}
                dist[(a, s)] = 1
                q = deque([(a, s)])
                while q:
                    v, p = q.popleft()
                    d = dist[(v, p)]
                    if d >= g - 1:
                        continue
                    for w in adj[v]:
                        if w == p:
                            continue
                        new_state = (w, v)
                        if new_state not in dist:
                            dist[new_state] = d + 1
                            if w == s and v != a:
                                if d + 1 < g:
                                    return True
                            q.append(new_state)
        return False

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def random_k_regular(n, k, seed=None):
        if n * k % 2 != 0:
            return None
        rng = random.Random(seed)
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        rng.shuffle(half_edges)
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in edges:
                return None
            edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        return sorted([list(e) for e in edges])

    def try_local_search(n_target, k, g, deadline):
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            edges = random_k_regular(n_target, k, seed=attempt)
            if edges is None:
                continue
            adj = build_adj(edges, n_target)
            if not nonbacktracking_girth(adj, g):
                return n_target, edges
            # If not girth g, try to improve it?
            # For now, just return if found.
        return None

    # Get LUW baseline
    n_luw, edges_luw = luw_construction()
    best_n, best_edges = n_luw, edges_luw

    # Try to find a smaller graph
    # Moore bound for (4,10) is 242.
    # Try n from 320 upwards? No, we want to minimize n.
    # But 320 is the known minimum.
    # If we can find 320, we win big.
    # If we can't, we return LUW (2048).
    
    # Try n=320
    if time.time() < deadline - 10:
        result = try_local_search(320, k, g, deadline)
        if result:
            best_n, best_edges = result

    # Try n=322, 324...
    for n_try in range(322, min(best_n, 400), 2):
        if time.time() > deadline - 10:
            break
        result = try_local_search(n_try, k, g, deadline)
        if result:
            n_found, edges_found = result
            if n_found < best_n:
                best_n, best_edges = n_found, edges_found
                break

    return best_n, best_edges
