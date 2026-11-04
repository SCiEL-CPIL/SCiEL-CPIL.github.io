import time
import random
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar construction
    def construct_lw(k, g):
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

    if k == 5 and g == 9:
        start_time = time.time()
        deadline = start_time + 140
        
        def min_dist(graph, u, v, max_depth):
            if u == v:
                return 0
            visited = {u: 0}
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                dist = visited[curr]
                if dist >= max_depth:
                    continue
                for nei in graph[curr]:
                    if nei == v:
                        return dist + 1
                    if nei not in visited:
                        visited[nei] = dist + 1
                        queue.append(nei)
            return -1

        def try_greedy(N, k, g, time_limit):
            graph = [set() for _ in range(N)]
            start_t = time.time()
            
            while True:
                if time.time() - start_t > time_limit:
                    return None
                
                candidates = [i for i in range(N) if len(graph[i]) < k]
                if not candidates:
                    return graph
                
                min_deg = min(len(graph[i]) for i in candidates)
                low_deg_candidates = [i for i in candidates if len(graph[i]) == min_deg]
                u = random.choice(low_deg_candidates)
                
                valid_v = []
                # Try random sampling first
                sample_size = min(50, len(candidates))
                sampled = random.sample(candidates, sample_size)
                for v in sampled:
                    if v == u or v in graph[u]:
                        continue
                    if min_dist(graph, u, v, g - 2) == -1:
                        valid_v.append(v)
                
                # If not found, do full scan
                if not valid_v:
                    for v in candidates:
                        if v == u or v in graph[u]:
                            continue
                        if min_dist(graph, u, v, g - 2) == -1:
                            valid_v.append(v)
                
                if not valid_v:
                    return None
                
                v = random.choice(valid_v)
                graph[u].add(v)
                graph[v].add(u)

        for N in [1100, 1150, 1200]:
            if time.time() > deadline:
                break
            for restart in range(10):
                if time.time() > deadline:
                    break
                result = try_greedy(N, k, g, 5.0)
                if result is not None:
                    edges = []
                    for i in range(N):
                        for j in result[i]:
                            if i < j:
                                edges.append((i, j))
                    return N, edges

        return construct_lw(k, g)
    else:
        return construct_lw(k, g)
