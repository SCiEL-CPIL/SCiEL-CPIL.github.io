import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def time_left():
        return deadline - time.time()

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

    def has_short_cycle(n, adj, g):
        half = (g - 1) // 2
        for start in range(n):
            parent = [-1] * n
            depth = [-1] * n
            parent[start] = -2
            depth[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if depth[node] >= half:
                    continue
                for nb in adj[node]:
                    if depth[nb] == -1:
                        depth[nb] = depth[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    elif parent[node] != nb and parent[nb] != node:
                        d1, d2 = depth[node], depth[nb]
                        u, v = node, nb
                        while u != v:
                            if d1 > d2:
                                u = parent[u]
                                d1 -= 1
                            elif d2 > d1:
                                v = parent[v]
                                d2 -= 1
                            else:
                                u = parent[u]
                                v = parent[v]
                                d1 -= 1
                                d2 -= 1
                        cycle_len = depth[node] + depth[nb] - 2 * d1 + 1
                        if cycle_len < g:
                            return True
        return False

    def random_regular_graph(n, k, rng):
        if n * k % 2 != 0:
            return None
        for _ in range(300):
            stubs = [i for i in range(n) for _ in range(k)]
            rng.shuffle(stubs)
            edges = []
            seen = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            if valid:
                return edges
        return None

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def try_edge_swap(adj, n, rng):
        for _ in range(100):
            a = rng.randint(0, n - 1)
            if not adj[a]:
                continue
            b = rng.choice(adj[a])
            c = rng.randint(0, n - 1)
            if c == a or c == b:
                continue
            if not adj[c]:
                continue
            d = rng.choice(adj[c])
            if d == a or d == b:
                continue
            if c in adj[a] or d in adj[b]:
                continue
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            return True
        return False

    def try_2swap(adj, n, rng):
        """Try a 2-edge swap: remove edges (a,b) and (c,d), add (a,c) and (b,d) or (a,d) and (b,c)"""
        for _ in range(200):
            a = rng.randint(0, n - 1)
            if len(adj[a]) < 2:
                continue
            b = rng.choice(adj[a])
            c = rng.randint(0, n - 1)
            if c == a or c == b:
                continue
            if len(adj[c]) < 2:
                continue
            d = rng.choice(adj[c])
            if d == a or d == b:
                continue
            # Check for parallel edges and self-loops
            if c in adj[a] or d in adj[b]:
                continue
            if a == c or a == d or b == c or b == d:
                continue
            # Check if adding (a,c) or (a,d) creates a parallel edge
            if c in adj[a]:
                continue
            if d in adj[b]:
                continue
            # Remove old edges
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            # Add new edges
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            return True
        return False

    def count_short_cycles(n, adj, g):
        """Count the number of cycles of length < g using BFS from each vertex"""
        half = (g - 1) // 2
        count = 0
        for start in range(n):
            parent = [-1] * n
            depth = [-1] * n
            parent[start] = -2
            depth[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if depth[node] >= half:
                    continue
                for nb in adj[node]:
                    if depth[nb] == -1:
                        depth[nb] = depth[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    elif parent[node] != nb and parent[nb] != node:
                        d1, d2 = depth[node], depth[nb]
                        u, v = node, nb
                        while u != v:
                            if d1 > d2:
                                u = parent[u]
                                d1 -= 1
                            elif d2 > d1:
                                v = parent[v]
                                d2 -= 1
                            else:
                                u = parent[u]
                                v = parent[v]
                                d1 -= 1
                                d2 -= 1
                        cycle_len = depth[node] + depth[nb] - 2 * d1 + 1
                        if cycle_len < g:
                            count += 1
        return count

    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Verify LUW construction has correct girth
    adj_luw = build_adj(n_luw, edges_luw)
    if has_short_cycle(n_luw, adj_luw, g):
        # Fallback: try to fix with local search
        rng = random.Random(42)
        for it in range(5000):
            if time_left() < 10:
                break
            if not try_edge_swap(adj_luw, n_luw, rng):
                continue
            if not has_short_cycle(n_luw, adj_luw, g):
                best_edges = [(min(u, v), max(u, v)) for u in range(n_luw) for v in adj_luw[u] if u < v]
                break
        else:
            # If still bad, return as-is (shouldn't happen for valid LUW)
            pass

    return best_n, best_edges
