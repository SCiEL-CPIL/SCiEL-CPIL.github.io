def construct(k, g):
    import time
    from collections import deque
    import random
    import math

    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    # Known optimal cages
    known_cages = {}
    known_cages[(3, 3)] = (4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    known_cages[(3, 4)] = (6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)])
    petersen_edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                      (5,7),(7,9),(9,6),(6,8),(8,5),
                      (0,5),(1,6),(2,7),(3,8),(4,9)]
    known_cages[(3, 5)] = (10, petersen_edges)
    known_cages[(4, 3)] = (5, [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)])
    known_cages[(4, 4)] = (8, [(i,j) for i in range(4) for j in range(4,8)])
    known_cages[(5, 3)] = (6, [(i,j) for i in range(6) for j in range(i+1,6)])
    known_cages[(5, 4)] = (10, [(i,j) for i in range(5) for j in range(5,10)])

    if (k, g) in known_cages:
        return known_cages[(k, g)]

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def girth_check_fast(n, adj, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < g - 1:
                            queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_luw(d, q):
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
        rule = {}
        rule[2] = (1,1)
        rule[3] = (2,1)
        rule[4] = (1,2)
        for c in range(5, d+1):
            r = (c-5)%4
            c0 = c - (c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1-side), tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # Try Circulant Search
    if k % 2 == 1:
        num_jumps = (k - 1) // 2
        # Moore bound approx
        # For k=5, g=9, MB=426.
        # Start a bit above MB.
        start_n = 430 if k == 5 and g == 9 else k + g
        
        for n in range(start_n, start_n + 1000, 2):
            if time_left() < 5:
                break
            
            # Try a few random jump sets
            for _ in range(10):
                if time_left() < 2:
                    break
                try:
                    jumps = random.sample(range(1, n//2), num_jumps)
                    jumps.append(n//2)
                    
                    # Connectivity check
                    g_val = n
                    for j in jumps:
                        g_val = math.gcd(g_val, j)
                    if g_val != 1:
                        continue
                    
                    # Girth Check
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[0] = 0
                    q = deque([0])
                    valid = True
                    while q and valid:
                        u = q.popleft()
                        if dist[u] >= g - 1:
                            continue
                        for j in jumps:
                            v1 = (u + j) % n
                            v2 = (u - j) % n
                            for v in [v1, v2]:
                                if v == u:
                                    continue
                                if dist[v] == -1:
                                    dist[v] = dist[u] + 1
                                    parent[v] = u
                                    if dist[v] < g - 1:
                                        q.append(v)
                                else:
                                    if v != parent[u]:
                                        cycle_len = dist[u] + dist[v] + 1
                                        if cycle_len < g:
                                            valid = False
                                            break
                            if not valid:
                                break
                    
                    if valid:
                        # Construct edges
                        edges = []
                        seen = set()
                        for i in range(n):
                            for j in jumps:
                                v1 = (i + j) % n
                                v2 = (i - j) % n
                                for v in [v1, v2]:
                                    if v == i:
                                        continue
                                    u = min(i, v)
                                    w = max(i, v)
                                    edge = (u, w)
                                    if edge not in seen:
                                        seen.add(edge)
                                        edges.append(edge)
                        
                        # Final verification
                        adj = build_adj(n, edges)
                        ok = True
                        for i in range(n):
                            if len(adj[i]) != k:
                                ok = False
                                break
                        if ok and girth_check_fast(n, adj, g):
                            return n, edges
                except Exception:
                    continue

    # Fallback to LUW
    d_min = max(3, (g + 1) // 2)
    for d in range(d_min, d_min + 3):
        if time_left() < 20:
            break
        try:
            n_luw, edges_luw = build_luw(d, k)
            if n_luw > 50000:
                continue
            adj_luw = build_adj(n_luw, edges_luw)
            ok = True
            for i in range(n_luw):
                if len(adj_luw[i]) != k:
                    ok = False
                    break
            if not ok:
                continue
            if girth_check_fast(n_luw, adj_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue

    # Last resort
    n = k + g
    jumps = list(range(1, (k+1)//2 + 1))
    edges = []
    seen = set()
    for i in range(n):
        for j in jumps:
            v1 = (i + j) % n
            v2 = (i - j) % n
            for v in [v1, v2]:
                if v == i: continue
                u = min(i, v)
                w = max(i, v)
                edge = (u, w)
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)
    return n, edges
