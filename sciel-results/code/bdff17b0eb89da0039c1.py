import time
import random

_DEADLINE = 110.0

def construct(k, g):
    _t0 = time.time()

    # LUW fallback for general cases
    def luw_construct(k, g):
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
            r = (c - 5) % 4
            c0 = c - r
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
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            if time.time() - _t0 > _DEADLINE:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if k != 4 or g != 10:
        return luw_construct(k, g)

    n_target = 320
    deadline = _t0 + _DEADLINE

    # Girth check using BFS, optimized
    def has_girth_at_least(n, adj, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    def random_regular_graph(n, k):
        # Configuration model with swaps to fix multi-edges and self-loops
        for _ in range(1000):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u != v:
                    edges.append((u, v))
            # Now fix multi-edges and self-loops via edge swaps
            valid = True
            edge_list = edges[:]
            attempts = 0
            while attempts < 100:
                # Find a problem: self-loop or multi-edge
                seen = set()
                problems = []
                for idx, (u, v) in enumerate(edge_list):
                    if u > v:
                        u, v = v, u
                    if u == v:
                        problems.append(idx)
                        break
                    if (u, v) in seen:
                        problems.append(idx)
                        break
                    seen.add((u, v))
                if not problems:
                    # Check all edges again fully
                    seen = set()
                    for u, v in edge_list:
                        a, b = (u, v) if u < v else (v, u)
                        if a == b or (a, b) in seen:
                            problems.append(0)
                            break
                        seen.add((a, b))
                    if not problems:
                        break
                # Pick a random problematic edge and another edge to swap
                idx1 = random.choice(problems)
                idx2 = random.randint(0, len(edge_list) - 1)
                if idx1 == idx2:
                    continue
                u1, v1 = edge_list[idx1]
                u2, v2 = edge_list[idx2]
                # Swap: (u1,v1) and (u2,v2) -> try (u1,v2) and (u2,v1)
                edge_list[idx1] = (u1, v2)
                edge_list[idx2] = (u2, v1)
                attempts += 1
            # Final check
            seen = set()
            for u, v in edge_list:
                a, b = (u, v) if u < v else (v, u)
                if a == b or (a, b) in seen:
                    break
                seen.add((a, b))
            else:
                # Build adjacency
                adj = [[] for _ in range(n)]
                for u, v in edge_list:
                    adj[u].append(v)
                    adj[v].append(u)
                return adj, edge_list
        return None, None

    for attempt in range(5000):
        if time.time() > deadline:
            break
        adj, edges = random_regular_graph(n_target, k)
        if edges and has_girth_at_least(n_target, adj, g):
            return n_target, edges

    return luw_construct(k, g)
