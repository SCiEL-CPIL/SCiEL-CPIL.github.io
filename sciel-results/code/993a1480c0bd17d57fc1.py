import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin

    # Fallback: LUW Graph
    def get_luw(k, g, force_d=None):
        q = k
        if force_d is not None:
            d = force_d
        else:
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

    def check_circulant_girth_fast(n, steps, g):
        # BFS on NB walk states (pos, last_step)
        # Start: (s, s) for each s in steps, dist=1
        # Goal: reach (0, s') with dist+1 < g
        
        visited = {}
        queue = []
        for s in steps:
            state = (s, s)
            visited[state] = 1
            queue.append(state)
        
        # We want to detect cycle length < g.
        # Cycle length = path_length + 1.
        # So we need path_length < g - 1.
        max_dist = g - 2
        
        head = 0
        while head < len(queue):
            u, last = queue[head]
            head += 1
            dist = visited[(u, last)]
            
            if dist >= max_dist:
                continue
                
            for s in steps:
                if s == (-last) % n:
                    continue
                v = (u + s) % n
                if v == 0:
                    # Cycle of length dist + 1
                    if dist + 1 < g:
                        return False
                else:
                    state = (v, s)
                    if state not in visited:
                        visited[state] = dist + 1
                        queue.append(state)
        return True

    def try_circulant(n, k, g):
        if k != 4:
            return None
        
        valid_steps = []
        for i in range(1, n // 2):
            valid_steps.append(i)
        
        # To ensure we cover all, we iterate through pairs
        # Number of pairs is ~ n^2/8. For n=320, ~12800.
        # Fast check allows this.
        
        for i in range(len(valid_steps)):
            if time.time() > deadline:
                return None
            s1 = valid_steps[i]
            for j in range(i + 1, len(valid_steps)):
                s2 = valid_steps[j]
                steps = [s1, n - s1, s2, n - s2]
                # Check distinctness
                if len(set(steps)) != 4:
                    continue
                
                if check_circulant_girth_fast(n, steps, g):
                    edges = []
                    for u in range(n):
                        for s in steps:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
        return None

    def check_graph_girth(n, edges, g):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check degree
        for u in range(n):
            if len(adj[u]) != k:
                return False
                
        # Check girth using NB walk BFS from 0
        # Since LUW is vertex transitive, checking from 0 is sufficient for girth?
        # Actually, for Cayley graphs, yes.
        steps = adj[0]
        return check_circulant_girth_fast(n, steps, g)

    if k == 4 and g == 10:
        # Try circulant graphs near the known optimum
        for n in [320, 322, 324, 326, 328, 330]:
            if time.time() > deadline:
                break
            res = try_circulant(n, k, g)
            if res:
                return res

        # Try LUW with smaller d first
        # d=4 -> 512 vertices. Check if girth is >= 10.
        if time.time() < deadline:
            n_luw, edges_luw = get_luw(k, g, force_d=4)
            if check_graph_girth(n_luw, edges_luw, g):
                return n_luw, edges_luw

        # Fallback to standard LUW (d=5 -> 2048 vertices)
        return get_luw(k, g)
    else:
        return get_luw(k, g)
