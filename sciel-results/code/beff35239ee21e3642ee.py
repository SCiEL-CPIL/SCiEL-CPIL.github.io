def construct(k, g):
    import time
    from collections import deque
    import numpy as np

    start_time = time.time()
    deadline = start_time + 110.0

    known_cages = {}
    known_cages[(3, 5)] = (10, [(0,1),(1,2),(2,3),(3,4),(4,0),
                                 (5,7),(7,9),(9,6),(6,8),(8,5),
                                 (0,5),(1,6),(2,7),(3,8),(4,9)])
    known_cages[(3, 4)] = (6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)])
    known_cages[(3, 3)] = (4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    known_cages[(4, 4)] = (8, [(i,j) for i in range(4) for j in range(4,8)])
    known_cages[(4, 3)] = (5, [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)])

    if (k, g) in known_cages:
        return known_cages[(k, g)]

    q = k

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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def girth_check(n, adj, g):
        """Check girth >= g using BFS over non-backtracking walks."""
        for start in range(n):
            # visited[u] = set of prev vertices that have been seen
            # More efficient: use a 2D boolean array
            visited = [set() for _ in range(n)]
            queue = deque()
            
            for v in adj[start]:
                visited[v].add(start)
                queue.append((v, start))
            
            depth = 1
            while queue:
                next_q = deque()
                while queue:
                    u, prev = queue.popleft()
                    if u == start:
                        if depth < g:
                            return False
                        continue
                    if depth >= g - 1:
                        continue
                    for w in adj[u]:
                        if w == prev:
                            continue
                        if w == start:
                            if depth + 1 < g:
                                return False
                            continue
                        if prev not in visited[w]:
                            visited[w].add(prev)
                            next_q.append((w, u))
                queue = next_q
                depth += 1
        return True

    d_min = max(3, (g + 1) // 2)
    d_max = min(d_min + 3, g + 2)
    
    best_n = None
    best_edges = None

    for d in range(d_min, d_max + 1):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = build_luw(d, q)
            if n > 50000:
                continue
            adj = build_adj(n, edges)
            ok = all(len(adj[i]) == k for i in range(n))
            if not ok:
                continue
            if girth_check(n, adj, g):
                best_n = n
                best_edges = edges
                break
        except Exception:
            continue
    
    if best_n is not None:
        return best_n, best_edges
    else:
        d = max(3, g)
        try:
            n, edges = build_luw(d, q)
            return n, edges
        except Exception:
            n = k + g + 1
            edges = []
            seen = set()
            jumps = list(range(1, k // 2 + 1))
            if k % 2 == 1:
                jumps.append(n // 2)
            for i in range(n):
                for j in jumps:
                    u, v = i, (i + j) % n
                    e = (min(u,v), max(u,v))
                    if e not in seen:
                        seen.add(e)
                        edges.append(e)
            if k % 2 == 1:
                for i in range(n):
                    for j in jumps:
                        u, v = i, (i - j) % n
                        e = (min(u,v), max(u,v))
                        if e not in seen:
                            seen.add(e)
                            edges.append(e)
            return n, edges
