import time
import random

def _check_girth_direct(n, edges, g):
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    max_depth = (g - 1) // 2
    for sv in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[sv] = 0
        queue = [sv]
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            if dist[v] >= max_depth:
                continue
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                elif w != parent[v]:
                    cycle_len = dist[v] + dist[w] + 1
                    if cycle_len < g:
                        return False
    return True

def _luw_construct(k, g, deadline):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y: (x+y)%q), (lambda x,y: (x-y)%q), (lambda x,y: x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    
    start = (0, (None,)+(0,)*d)
    index, order, edges = {start:0}, [start], []
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100
    
    if k == 3 and g == 16:
        base_edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                      (5,7),(7,9),(9,6),(6,8),(8,5),
                      (0,5),(1,6),(2,7),(3,8),(4,9)]
        n_base = 10
        
        for n in range(8, 200):
            if time.time() > deadline:
                break
            n_lift = n_base * n
            if n_lift > 500:
                break
            for attempt in range(1000):
                if time.time() > deadline:
                    break
                volt = {}
                for (u,v) in base_edges:
                    val = random.randint(1, n-1)
                    volt[(u,v)] = val
                    volt[(v,u)] = (-val) % n
                
                edges = []
                for (u,v) in base_edges:
                    vu = volt[(u,v)]
                    for h in range(n):
                        a = u * n + h
                        b = v * n + (h + vu) % n
                        if a < b:
                            edges.append((a,b))
                        else:
                            edges.append((b,a))
                
                if _check_girth_direct(n_lift, edges, g):
                    return n_lift, edges
        
        return _luw_construct(k, g, deadline)
    else:
        return _luw_construct(k, g, deadline)
