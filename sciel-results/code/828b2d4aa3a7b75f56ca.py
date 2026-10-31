import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k == 3 and g == 16:
        # Base graph: Cube Q3
        # Vertices 0..7
        base_edges = []
        for u in range(8):
            for v in range(u + 1, 8):
                if bin(u ^ v).count('1') == 1:
                    base_edges.append((u, v))
        
        group_size = 117
        n_lifted = 8 * group_size
        
        # Helper to check girth
        def check_girth(n, edges, target_girth):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check degree
            for i in range(n):
                if len(adj[i]) != k:
                    return False
            
            # BFS from each vertex
            max_depth = target_girth // 2
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            if v != parent[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < target_girth:
                                    return False
            return True

        # Random voltage search
        for attempt in range(5000):
            if time.time() > deadline:
                break
                
            voltages = {}
            for u, v in base_edges:
                vu = random.randint(0, group_size - 1)
                voltages[(u, v)] = vu
                voltages[(v, u)] = (-vu) % group_size
            
            lifted_edges = []
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for x in range(group_size):
                    y = (x + vu) % group_size
                    u_lift = u * group_size + x
                    v_lift = v * group_size + y
                    lifted_edges.append((u_lift, v_lift))
            
            if check_girth(n_lifted, lifted_edges, g):
                return n_lifted, lifted_edges
        
        # Fallback to LUW
        return _seed_construct(k, g)
        
    else:
        return _seed_construct(k, g)

def _seed_construct(k, g):
    # Original seed code
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
