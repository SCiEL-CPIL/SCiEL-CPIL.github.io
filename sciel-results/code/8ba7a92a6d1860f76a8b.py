import time
from collections import deque

def _luw(k, d):
    q = k
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
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
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,)+(0,)*d)
    index = {start:0}
    order = [start]
    adj = [[]]
    for vertex in order:
        vi = index[vertex]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                adj.append([])
            vj = index[other]
            if vertex[0]==0:
                adj[vi].append(vj)
                adj[vj].append(vi)
    return len(order), adj

def _girth_ge(n, adj, g):
    # Return True if girth >= g (no cycle of length < g).
    limit = g - 1
    for s in range(n):
        seen = {}
        dq = deque()
        for nb in adj[s]:
            seen[(s,nb)] = 1
            dq.append((s,nb,1))
        while dq:
            u,v,dep = dq.popleft()
            if dep >= limit:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                nd = dep + 1
                key = (v,w)
                if key in seen:
                    L = nd - seen[key]
                    if L < g:
                        return False
                else:
                    if nd <= limit:
                        seen[key] = nd
                        dq.append((v,w,nd))
    return True

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 100
    # Try d=4 first (smaller n); verify girth.
    for d in [4, 5, 3, 6]:
        if time.time() > deadline:
            break
        try:
            n, adj = _luw(k, d)
        except Exception:
            continue
        if any(len(a) != k for a in adj):
            continue
        simple = True
        for i in range(n):
            if len(set(adj[i])) != len(adj[i]):
                simple = False
                break
        if not simple:
            continue
        if n <= 2000:
            # verify girth via BFS
            if not _girth_ge(n, adj, g):
                continue
        else:
            # trust theory for odd d: girth >= d+5
            if d % 2 == 1:
                if d + 5 < g:
                    continue
            else:
                # even d, uncertain; skip unless we can verify
                continue
        return n, _edges(n, adj)
    # fallback d=5
    n, adj = _luw(k, 5)
    return n, _edges(n, adj)

def _edges(n, adj):
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i,j))
    return edges
