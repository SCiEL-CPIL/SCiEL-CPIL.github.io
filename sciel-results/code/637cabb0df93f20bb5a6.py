import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    def luw_construct():
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

    if k != 5 or g != 9:
        return luw_construct()

    n_min = 426
    
    def find_shortest_cycle(adj, n):
        min_len = 10**9
        for start in range(n):
            visited = {}
            queue = []
            for nb in adj[start]:
                queue.append((start, nb, 1))
                visited[(start, nb)] = 1
            
            while queue:
                u, p, dist = queue.pop(0)
                if dist >= min_len:
                    continue
                for v in adj[u]:
                    if v == p:
                        continue
                    if v == start:
                        min_len = min(min_len, dist + 1)
                        break
                    state = (u, v)
                    if state not in visited:
                        visited[state] = dist + 1
                        queue.append((u, v, dist + 1))
        return min_len if min_len < 10**9 else 10**9

    def random_5_reg(n):
        adj = [[] for _ in range(n)]
        edges = []
        attempts = 0
        while len(edges) < n * k // 2:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if v in adj[u]: continue
            if len(adj[u]) >= k or len(adj[v]) >= k: continue
            
            adj[u].append(v)
            adj[v].append(u)
            edges.append((u, v))
            attempts += 1
            if attempts > 50000:
                return None
        return adj

    best_n = None
    best_edges = None
    
    for n in range(n_min, 1500):
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            adj = random_5_reg(n)
            if adj is None:
                continue
                
            curr_girth = find_shortest_cycle(adj, n)
            if curr_girth >= g:
                found = True
                break
                
            for step in range(2000):
                if time.time() > deadline:
                    break
                    
                if curr_girth >= g:
                    found = True
                    break
                    
                u = random.randint(0, n-1)
                v = random.choice(adj[u])
                w = random.randint(0, n-1)
                x = random.choice(adj[w])
                
                if u == w or u == x or v == w or v == x:
                    continue
                if w in adj[u] or x in adj[u] or w in adj[v] or x in adj[v]:
                    continue
                    
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                
                adj[u].append(w)
                adj[w].append(u)
                adj[v].append(x)
                adj[x].append(v)
                
                new_girth = find_shortest_cycle(adj, n)
                
                if new_girth <= curr_girth:
                    curr_girth = new_girth
                else:
                    adj[u].remove(w)
                    adj[w].remove(u)
                    adj[v].remove(x)
                    adj[x].remove(v)
                    
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[w].append(x)
                    adj[x].append(w)
                    
        if found:
            best_n = n
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            break

    if best_n is not None:
        return best_n, best_edges
    else:
        return luw_construct()
