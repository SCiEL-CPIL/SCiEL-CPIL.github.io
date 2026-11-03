import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 115
    
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
    
    if k != 4:
        return luw_construct(k, g)
    
    def full_girth_check(adj, n, g):
        depth_limit = (g - 1) // 2
        for start in range(n):
            depth = [-1] * n
            parent = [-1] * n
            depth[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if depth[u] >= depth_limit:
                    continue
                for v in adj[u]:
                    if depth[v] == -1:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and depth[v] <= depth[u]:
                        a, b = u, v
                        da, db = depth[a], depth[b]
                        while da > db:
                            a = parent[a]
                            da -= 1
                        while db > da:
                            b = parent[b]
                            db -= 1
                        while a != b:
                            a = parent[a]
                            b = parent[b]
                        lca_depth = depth[a]
                        cycle_len = depth[u] + depth[v] - 2 * lca_depth + 1
                        if cycle_len < g:
                            return False
        return True
    
    def fast_girth_check(adj, n, g, sample_size=10):
        depth_limit = (g - 1) // 2
        starts = random.sample(range(n), min(sample_size, n))
        for start in starts:
            depth = [-1] * n
            parent = [-1] * n
            depth[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if depth[u] >= depth_limit:
                    continue
                for v in adj[u]:
                    if depth[v] == -1:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and depth[v] <= depth[u]:
                        a, b = u, v
                        da, db = depth[a], depth[b]
                        while da > db:
                            a = parent[a]
                            da -= 1
                        while db > da:
                            b = parent[b]
                            db -= 1
                        while a != b:
                            a = parent[a]
                            b = parent[b]
                        lca_depth = depth[a]
                        cycle_len = depth[u] + depth[v] - 2 * lca_depth + 1
                        if cycle_len < g:
                            return False
        return True
    
    def random_4_regular(n):
        for _ in range(200):
            stubs = [i for i in range(n) for _ in range(4)]
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            i = 0
            while i < len(stubs):
                a, b = stubs[i], stubs[i+1]
                i += 2
                if a == b or b in adj[a]:
                    valid = False
                    break
                adj[a].add(b)
                adj[b].add(a)
            if valid and all(len(adj[v]) == 4 for v in range(n)):
                return adj
        return None
    
    n_values = [162, 164, 166, 170, 180, 200, 250, 300, 320, 350, 400]
    for n in n_values:
        if time.time() - start_time > time_limit:
            break
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            adj = random_4_regular(n)
            if adj is None:
                continue
            # Local search
            for _ in range(500):
                if time.time() - start_time > time_limit:
                    break
                if fast_girth_check(adj, n, g):
                    break
                # Pick two random edges
                all_edges = [(v, u) for v in range(n) for u in adj[v] if v < u]
                if len(all_edges) < 2:
                    break
                idx1 = random.randint(0, len(all_edges) - 1)
                idx2 = random.randint(0, len(all_edges) - 1)
                if idx1 == idx2:
                    continue
                e1, e2 = all_edges[idx1], all_edges[idx2]
                a, b = e1
                c, d = e2
                for variant in [0, 1]:
                    if variant == 0:
                        new_e1, new_e2 = (a, c), (b, d)
                    else:
                        new_e1, new_e2 = (a, d), (b, c)
                    valid = True
                    for x, y in [new_e1, new_e2]:
                        if x == y or y in adj[x]:
                            valid = False
                            break
                    if not valid:
                        continue
                    adj_copy = [set(s) for s in adj]
                    adj_copy[a].remove(b)
                    adj_copy[b].remove(a)
                    adj_copy[c].remove(d)
                    adj_copy[d].remove(c)
                    for x, y in [new_e1, new_e2]:
                        adj_copy[x].add(y)
                        adj_copy[y].add(x)
                    if fast_girth_check(adj_copy, n, g):
                        adj = adj_copy
                        break
            # Full check
            if full_girth_check(adj, n, g):
                edges = [(v, u) for v in range(n) for u in adj[v] if v < u]
                return n, edges
    
    return luw_construct(k, g)
