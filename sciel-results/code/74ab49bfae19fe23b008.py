import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # LUW Construction
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

    n_luw, edges_luw = luw_construct(k, g)

    # Fast girth check
    def check_girth(adj, n, target_girth):
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] + 1 >= target_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    def get_girth(adj, n, max_check=20):
        girth = max_check
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= max_check // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
                            if girth == 3:
                                return 3
        return girth

    def random_regular_graph(n, k, max_attempts=10):
        if n * k % 2 != 0:
            return None
        for _ in range(max_attempts):
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                return adj
        return None

    def swap_edges(adj, n, k):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        if len(edges) < 2:
            return False
        idx1 = random.randint(0, len(edges) - 1)
        idx2 = random.randint(0, len(edges) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(edges) - 1)
        u1, v1 = edges[idx1]
        u2, v2 = edges[idx2]
        options = [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]
        random.shuffle(options)
        for (a1, b1), (a2, b2) in options:
            if a1 == b1 or a2 == b2:
                continue
            if b1 in adj[a1] or b2 in adj[a2]:
                continue
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            adj[u2].discard(v2)
            adj[v2].discard(u2)
            adj[a1].add(b1)
            adj[b1].add(a1)
            adj[a2].add(b2)
            adj[b2].add(a2)
            return True
        return False

    # Moore bound
    if g % 2 == 0:
        r = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    else:
        r = (g - 1) // 2
        moore_bound = k * sum((k - 1) ** i for i in range(r)) + (k - 1) ** r

    n_start = moore_bound
    if k % 2 == 1 and n_start % 2 != 0:
        n_start += 1

    # Try local search for smaller graphs
    max_n_to_try = min(n_luw, n_start + 50)
    for n in range(n_start, max_n_to_try + 1):
        if time.time() > deadline - 15.0:
            break
        if k % 2 == 1 and n % 2 != 0:
            continue

        for restart in range(2):
            if time.time() > deadline - 15.0:
                break
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
            if check_girth(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges

            current_girth = get_girth(adj, n, max_check=g)
            T, cooling, min_T = 1.0, 0.995, 0.05
            for _ in range(300):
                if time.time() > deadline - 15.0:
                    break
                if not swap_edges(adj, n, k):
                    continue
                new_girth = get_girth(adj, n, max_check=g)
                delta = new_girth - current_girth
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_girth = new_girth
                    if current_girth >= g:
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        return n, edges
                T *= cooling
                if T < min_T:
                    break

    return n_luw, edges_luw
