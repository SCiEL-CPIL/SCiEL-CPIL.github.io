import random
import time
from collections import deque

def construct(k, g):
    # Voltage graph construction with Petersen graph as base
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    base_n = 10
    start_time = time.time()
    
    for m in range(94, 130):
        n = base_n * m
        
        for attempt in range(300):
            if time.time() - start_time > 100:
                break
            
            voltages = [random.randint(0, m - 1) for _ in range(len(petersen_edges))]
            
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for e_idx, (u, v) in enumerate(petersen_edges):
                vol = voltages[e_idx]
                for i in range(m):
                    u_lift = u * m + i
                    v_lift = v * m + (i + vol) % m
                    adj[u_lift].append(v_lift)
                    adj[v_lift].append(u_lift)
            
            # Quick girth check: BFS from several vertices, depth g-1
            def check_from(adj, start_v, g):
                q = deque()
                visited = set()
                q.append((start_v, -1, 0))
                visited.add((start_v, -1))
                
                while q:
                    v, p, d = q.popleft()
                    if d >= g - 1:
                        continue
                    
                    for w in adj[v]:
                        if w == p:
                            continue
                        if w == start_v:
                            return d + 1
                        state = (w, v)
                        if state not in visited:
                            visited.add(state)
                            q.append((w, v, d + 1))
                
                return g
            
            all_good = True
            for check_v in [0, 1, n // 3, n // 2, 2 * n // 3, n - 1]:
                if check_from(adj, check_v, g) < g:
                    all_good = False
                    break
            
            if all_good:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
        
        if time.time() - start_time > 100:
            break
    
    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = None
            y[1] = first
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
    queue = deque([start])
    
    while queue:
        vertex = queue.popleft()
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
