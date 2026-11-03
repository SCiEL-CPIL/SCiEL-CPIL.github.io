import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Moore bound
    n_min = 2 * k * ((k-1)**(g//2 - 1) - 1) // (k-2) if k > 2 else g
    if k == 3:
        n_min = 512
        
    n_max = 1200
    
    def gen_random_k_reg(n, k):
        # Configuration model with repair
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        
        random.shuffle(half_edges)
        
        edges = []
        for i in range(0, len(half_edges), 2):
            edges.append([half_edges[i], half_edges[i+1]])
        
        # Repair
        for _ in range(100):
            # Find a bad edge
            bad_idx = -1
            seen = set()
            for i, (u, v) in enumerate(edges):
                if u == v:
                    bad_idx = i
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in seen:
                    bad_idx = i
                    break
                seen.add((u, v))
            
            if bad_idx == -1:
                return [(u, v) for u, v in edges]
            
            # Swap
            u1, v1 = edges[bad_idx]
            for j in range(len(edges)):
                if j == bad_idx: continue
                u2, v2 = edges[j]
                # Swap v1 and v2
                edges[bad_idx] = [u1, v2]
                edges[j] = [u2, v1]
                break
        return None

    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            
            while queue:
                u = queue.pop(0)
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    for n in range(n_min, n_max + 1):
        if time.time() > deadline:
            break
        for _ in range(5):
            if time.time() > deadline:
                break
            edges = gen_random_k_reg(n, k)
            if edges and check_girth(n, edges, g):
                return n, edges
                
    # Fallback LUW
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
