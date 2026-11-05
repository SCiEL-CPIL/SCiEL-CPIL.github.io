import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = None
    best_edges = None
    
    for n in range(512, 1000, 2):
        if time.time() > deadline:
            break
        for a in range(1, n // 3 + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, n // 2):
                if time.time() > deadline:
                    break
                c = (n - a - b) % n
                if c <= b or c == 0:
                    continue
                # Check distinctness
                if a == b or a == c or b == c:
                    continue
                # BFS from vertex 0
                dist = [-1] * n
                parent = [-1] * n
                dist[0] = 0
                q = deque([0])
                girth = float('inf')
                
                while q:
                    u = q.popleft()
                    if dist[u] >= g:
                        continue
                    for v in [(u + a) % n, (u + b) % n, (u + c) % n]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u] and dist[v] <= dist[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth:
                                girth = cycle_len
                            if girth < g:
                                break
                    if girth < g:
                        break
                
                if girth >= g:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = []
                        for i in range(n):
                            for j in [(i + a) % n, (i + b) % n, (i + c) % n]:
                                if i < j:
                                    best_edges.append((i, j))
                        if best_n < 936:
                            return best_n, best_edges
        if best_n is not None and n > best_n:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: LUW
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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
