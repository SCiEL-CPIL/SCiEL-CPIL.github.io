import time
import random
import sys
from collections import defaultdict, deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound: minimum possible n for a k-regular graph of girth g
    moore_bound = 2 + k * sum((k - 1) ** i for i in range((g - 1) // 2))

    # LUW (Lazebnik-Ustimenko-Woldar) construction
    def get_luw_graph(q, d):
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if time_left() < 5:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Fast girth verification using adjacency matrix
    def verify_girth_fast(n, edges, g_limit, k_expected):
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        # Check regularity
        degs = adj.sum(axis=1)
        if np.any(degs != k_expected):
            return False
        # Check girth via BFS from each vertex (using numpy for speed)
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                v = queue.popleft()
                neighbors = np.where(adj[v] == 1)[0]
                for u in neighbors:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        parent[u] = v
                        queue.append(u)
                    elif u != parent[v] and v != parent[u]:
                        cycle_len = int(dist[v] + dist[u] + 1)
                        if cycle_len < g_limit:
                            return False
        return True

    # Try LUW with different d values to find smallest valid graph
    best_n = None
    best_edges = None

    # The LUW graph CD(d,q) has girth at least 2d+4 (for appropriate d)
    # We need girth >= g, so 2d+4 >= g => d >= (g-4)/2
    # But the seed uses a different formula. Let's try the seed's formula first,
    # then try smaller d values to see if they still give girth >= g.
    
    # Seed formula: d = max(3, g-5 if (g-5)%2 else g-4)
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Try q = k with seed d first
    luw_n, luw_edges = get_luw_graph(k, d_seed)
    
    if verify_girth_fast(luw_n, luw_edges, g, k):
        best_n = luw_n
        best_edges = luw_edges
    else:
        # Try other d values
        for d_try in range(max(3, (g-4)//2), d_seed + 1):
            if time_left() < 10:
                break
            n_try, e_try = get_luw_graph(k, d_try)
            if verify_girth_fast(n_try, e_try, g, k):
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = e_try
    
    # If no valid LUW found, try to construct one via random + repair
    if best_n is None:
        # Fallback: generate random k-regular graph and repair
        def gen_random_reg(n, k):
            if n * k % 2 != 0:
                return None, None
            half_edges = [i for i in range(n) for _ in range(k)]
            random.shuffle(half_edges)
            adj = [set() for _ in range(n)]
            edges = []
            for i in range(0, n * k, 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v or v in adj[u]:
                    return None, None
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
            return adj, edges

        def count_short_cycles(adj, n, g_limit):
            count = 0
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    v = queue.popleft()
                    for u in adj[v]:
                        if dist[u] == -1:
                            dist[u] = dist[v] + 1
                            parent[u] = v
                            queue.append(u)
                        elif u != parent[v] and v != parent[u]:
                            cycle_len = dist[v] + dist[u] + 1
                            if cycle_len < g_limit:
                                count += 1
            return count

        def try_swap(adj, n, k, edges):
            if len(edges) < 2:
                return False, None
            for _ in range(100):
                i = random.randint(0, len(edges) - 1)
                j = random.randint(0, len(edges) - 1)
                if i == j:
                    continue
                u, v = edges[i]
                x, y = edges[j]
                if len({u, v, x, y}) < 4:
                    continue
                if x in adj[u] or y in adj[v]:
                    continue
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
                edges[i] = (u, x)
                edges[j] = (v, y)
                return True, (i, j, u, v, x, y)
            return False, None

        def undo_swap(adj, edges, swap_info):
            i, j, u, v, x, y = swap_info
            adj[u].remove(x)
            adj[x].remove(u)
            adj[v].remove(y)
            adj[y].remove(v)
            adj[u].add(v)
            adj[v].add(u)
            adj[x].add(y)
            adj[y].add(x)
            edges[i] = (u, v)
            edges[j] = (x, y)

        # Try several target sizes
        offsets = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
        for offset in offsets:
            if time_left() < 15:
                break
            target_n = moore_bound + offset
            if target_n < 2 * k:
                continue
            if target_n * k % 2 != 0:
                continue

            for attempt in range(5):
                if time_left() < 15:
                    break

                adj, edges = gen_random_reg(target_n, k)
                if adj is None:
                    continue

                violations = count_short_cycles(adj, target_n, g)
                if violations == 0:
                    best_n = target_n
                    best_edges = edges
                    break

                T = 50.0
                cooling = 0.9995
                min_T = 0.1
                max_steps = 3000

                for step in range(max_steps):
                    if time_left() < 15:
                        break
                    if T < min_T:
                        break

                    old_violations = violations
                    success, swap_info = try_swap(adj, target_n, k, edges)
                    if not success:
                        continue

                    new_violations = count_short_cycles(adj, target_n, g)
                    delta = new_violations - old_violations

                    if delta < 0 or random.random() < (10 ** (-delta / T)):
                        violations = new_violations
                        if violations == 0:
                            best_n = target_n
                            best_edges = edges
                            break
                    else:
                        undo_swap(adj, edges, swap_info)

                    T *= cooling

                if best_n is not None:
                    break

            if best_n is not None:
                break

    # Final safety: if still no valid graph, return LUW with seed d (best effort)
    if best_n is None:
        best_n, best_edges = get_luw_graph(k, d_seed)

    return best_n, best_edges
