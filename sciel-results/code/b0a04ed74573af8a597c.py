import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate LUW graph (Seed)
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
                
    best_n = len(order)
    best_edges = edges_luw
    
    # 2. Moore Bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        s = 0
        for i in range(r):
            s += k * ((k-1) ** i)
        moore = 2 * s
        
    # 3. Random Search for smaller n
    def get_random_k_regular(n):
        if n * k % 2 != 0 or n < k + 1:
            return None
        adj = [set() for _ in range(n)]
        # Cycle
        for i in range(n):
            j = (i + 1) % n
            adj[i].add(j)
            adj[j].add(i)
        remaining = [k - 2] * n
        need = [i for i in range(n) if remaining[i] > 0]
        
        # Retry logic inside
        for _ in range(10):
            need = [i for i in range(n) if remaining[i] > 0]
            if not need:
                break
            random.shuffle(need)
            success = True
            for u in need:
                if remaining[u] == 0: continue
                candidates = [v for v in range(n) if v != u and v not in adj[u] and remaining[v] > 0]
                if not candidates:
                    success = False
                    break
                # Choose v
                v = random.choice(candidates)
                adj[u].add(v)
                adj[v].add(u)
                remaining[u] -= 1
                remaining[v] -= 1
            if success:
                break
        if not success:
            return None
            
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    def check_girth(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        max_depth = (g - 1) // 2
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = deque([(start, 0)])
            visited[start] = 0
            while queue:
                node, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for nei in adj[node]:
                    if visited[nei] == -1:
                        visited[nei] = depth + 1
                        parent[nei] = node
                        queue.append((nei, depth + 1))
                    elif nei != parent[node]:
                        cycle_len = depth + 1 + visited[nei]
                        if cycle_len < g:
                            return False
        return True

    current_n = moore
    while time.time() < deadline and current_n < best_n:
        found = False
        attempts = 50
        for _ in range(attempts):
            if time.time() > deadline: break
            edges = get_random_k_regular(current_n)
            if edges is None:
                continue
            if check_girth(edges, current_n, g):
                best_n = current_n
                best_edges = edges
                found = True
                break
        if found:
            # We found one, try to go smaller?
            # We can't easily go smaller than moore.
            # If we found one at moore, we are done (can't get smaller).
            # If we found one at moore+10, we can try moore+1...
            # But we are iterating current_n upwards.
            # So if we found one, we should break and return?
            # No, we want the MINIMUM n.
            # If we found one at current_n, we should check if we can find one at current_n - 1?
            # We are iterating from moore upwards.
            # So if we found one at moore, it's the best possible.
            # If we failed at moore, we go to moore+1.
            # If we succeed at moore+1, we return it?
            # Yes, because we are checking in increasing order of n.
            break
        else:
            current_n += 1

    return best_n, best_edges
