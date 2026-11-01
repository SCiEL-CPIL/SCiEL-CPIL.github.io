import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def time_ok():
        return time.time() < deadline

    # Moore bound for k-regular, girth g
    if g % 2 == 0:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    else:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))

    # Non-backtracking BFS to find girth
    def compute_girth(n, adj):
        min_girth = float('inf')
        for s in range(n):
            visited = set()
            queue = deque()
            for nb in adj[s]:
                visited.add((s, nb))
                queue.append((nb, s, 1))
            while queue:
                v, prev, dist = queue.popleft()
                if dist >= min_girth:
                    continue
                for nb in adj[v]:
                    if nb == prev:
                        continue
                    if nb == s:
                        if dist + 1 < min_girth:
                            min_girth = dist + 1
                        continue
                    if (v, nb) not in visited:
                        visited.add((v, nb))
                        queue.append((nb, v, dist + 1))
            if min_girth <= g:
                break
        return min_girth

    def random_3reg(n):
        if n % 2 != 0:
            return None
        half = n * 3
        if half % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            stubs.extend([i] * 3)
        random.shuffle(stubs)
        edges = []
        for i in range(0, half, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None
            edges.append((u, v))
        edge_set = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != 3:
                return None
        return edges

    # Circulant search for C_n(1, a, b)
    def _has_short_cycle_circulant(n, a, b, g):
        visited = [-1] * n
        parent = [-1] * n
        visited[0] = 0
        queue = deque([(0, 0)])
        max_depth = g // 2
        while queue:
            u, depth = queue.popleft()
            for shift in [1, a, b]:
                for v in [(u + shift) % n, (u - shift) % n]:
                    if visited[v] == -1:
                        visited[v] = depth + 1
                        parent[v] = u
                        if depth + 1 <= max_depth:
                            queue.append((v, depth + 1))
                    else:
                        if parent[v] == u or parent[u] == v:
                            continue
                        cycle_len = depth + 1 + visited[v]
                        if cycle_len < g:
                            return True
        return False

    def _find_circulant_cage(k, g, deadline):
        min_n = max(moore, 512)
        if min_n % 2 != 0:
            min_n += 1
        max_n = min_n + 400
        for n in range(min_n, max_n + 1, 2):
            if time.time() > deadline - 5:
                return None, None
            for a in range(2, n // 2):
                if time.time() > deadline - 5:
                    return None, None
                for b in range(a + 1, n // 2):
                    if _has_short_cycle_circulant(n, a, b, g):
                        continue
                    edges = []
                    for i in range(n):
                        for shift in [1, a, b]:
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        return None, None

    # LUW construction
    def _luw_construct_with_d(k, g, d):
        q = k
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

    # Edge swap for local search
    def edge_swap(edges, n):
        if len(edges) < 2:
            return None
        for _ in range(30):
            i = random.randint(0, len(edges) - 1)
            j = random.randint(0, len(edges) - 1)
            if i == j:
                continue
            a, b = edges[i]
            c, d = edges[j]
            if len({a, b, c, d}) < 4:
                continue
            new_edges = list(edges)
            del new_edges[j]
            del new_edges[i]
            existing = set((min(u, v), max(u, v)) for u, v in new_edges)
            key1 = (min(a, c), max(a, c))
            key2 = (min(b, d), max(b, d))
            if key1 in existing or key2 in existing:
                continue
            new_edges.append((a, c))
            new_edges.append((b, d))
            deg = [0] * n
            for u, v in new_edges:
                deg[u] += 1
                deg[v] += 1
            if all(d == 3 for d in deg):
                return new_edges
        return None

    # Strategy: try circulant first (fast, structured), then random+local search, then LUW
    best_n = None
    best_edges = None

    # 1. Try circulant search
    try:
        n_circ, edges_circ = _find_circulant_cage(k, g, deadline)
        if n_circ is not None:
            best_n = n_circ
            best_edges = edges_circ
    except Exception:
        pass

    if best_n is not None:
        return best_n, best_edges

    # 2. Try random 3-regular with local search
    n_start = max(moore, 512)
    if n_start % 2 != 0:
        n_start += 1

    n_trial = n_start
    max_n = n_start + 300

    while n_trial <= max_n and time_ok():
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts and time_ok():
            edges = random_3reg(n_trial)
            if edges is None:
                attempts += 1
                continue

            adj = [[] for _ in range(n_trial)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            girth = compute_girth(n_trial, adj)
            if girth >= g:
                best_n = n_trial
                best_edges = edges
                break

            # Local search if girth is close
            if girth >= g - 2 and time_ok():
                for _ in range(50):
                    if not time_ok():
                        break
                    new_edges = edge_swap(edges, n_trial)
                    if new_edges is None:
                        continue
                    new_adj = [[] for _ in range(n_trial)]
                    for u, v in new_edges:
                        new_adj[u].append(v)
                        new_adj[v].append(u)
                    new_girth = compute_girth(n_trial, new_adj)
                    if new_girth >= girth:
                        edges = new_edges
                        adj = new_adj
                        girth = new_girth
                        if girth >= g:
                            best_n = n_trial
                            best_edges = edges
                            break
            attempts += 1

        if best_n is not None:
            break
        n_trial += 2

    if best_n is not None:
        return best_n, best_edges

    # 3. Fallback: LUW construction
    for d in [7, 9, 11]:
        if not time_ok():
            break
        try:
            n_luw, edges_luw = _luw_construct_with_d(k, g, d)
            adj = [[] for _ in range(n_luw)]
            for u, v in edges_luw:
                adj[u].append(v)
                adj[v].append(u)
            girth = compute_girth(n_luw, adj)
            if girth >= g:
                return n_luw, edges_luw
        except Exception:
            pass

    # Final fallback: LUW with default d
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    n_luw, edges_luw = _luw_construct_with_d(k, g, d)
    return n_luw, edges_luw
