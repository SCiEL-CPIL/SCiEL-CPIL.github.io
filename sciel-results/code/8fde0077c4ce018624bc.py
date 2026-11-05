import time, math, random
from collections import deque

def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar: guaranteed valid fallback.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y:(x+y)%q
        sub = lambda x,y:(x-y)%q
        mul = lambda x,y: x*y%q
    rule = {2:(1,1),3:(2,1),4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
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
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0]==0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _girth_ok(adj, n, g):
    # Exact girth certification via non-backtracking BFS from every vertex.
    max_depth = (g-1)//2
    for s in range(n):
        dist = [-1]*n
        par = [-1]*n
        dist[s]=0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            for v in adj[u]:
                if v == par[u]:
                    continue
                dv = dist[v]
                if dv == -1:
                    if du+1 <= max_depth:
                        dist[v]=du+1; par[v]=u; dq.append(v)
                elif du+dv+1 < g:
                    return False
    return True

def _short_path(adj, n, u, v, maxd):
    # True iff dist(u,v) <= maxd in the current graph.
    if u==v:
        return 0<=maxd
    seen = {u:0}
    dq = deque([u])
    while dq:
        x = dq.popleft()
        d = seen[x]
        if d >= maxd:
            continue
        for y in adj[x]:
            if y==v:
                return True
            if y not in seen:
                seen[y]=d+1
                dq.append(y)
    return False

def _greedy(n, k, g, rng, deadline):
    # Add only edges that cannot create a cycle shorter than g.
    adj = [[] for _ in range(n)]
    eset = set()
    deg = [0]*n
    target = n*k//2
    added = 0
    maxd = g-2
    while added < target:
        if time.time() > deadline:
            return None
        progress = False
        for _ in range(4000):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u==v or deg[u]>=k or deg[v]>=k:
                continue
            key = (u,v) if u<v else (v,u)
            if key in eset:
                continue
            if _short_path(adj, n, u, v, maxd):
                continue
            adj[u].append(v); adj[v].append(u)
            eset.add(key)
            deg[u]+=1; deg[v]+=1
            added+=1
            progress=True
            break
        if not progress:
            return None
    return adj

def _random_k_regular(n, k, rng, tries=300):
    if (n*k)%2:
        return None
    stubs = []
    for v in range(n):
        stubs.extend([v]*k)
    for _ in range(tries):
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        ok=True
        for i in range(0, n*k, 2):
            a,b = stubs[i], stubs[i+1]
            if a==b or b in adj[a]:
                ok=False; break
            adj[a].append(b); adj[b].append(a)
        if ok:
            return adj
    return None

def _edges(adj, n):
    e=[]
    for i in range(n):
        for j in adj[i]:
            if i<j:
                e.append((i,j))
    return e

def construct(k, g):
    start = time.time()
    deadline = start + 118.0
    if g % 2 == 1:
        r = (g-3)//2
        moore = k * sum((k-1)**i for i in range(r+1))
    else:
        r = g//2 - 1
        moore = k*(k-1) * sum((k-1)**i for i in range(r))
    n0 = int(math.ceil(moore))
    if n0 < 2: n0 = 2
    if (n0*k)%2: n0 += 1
    max_vertices = 1200
    rng = random.Random(12345)
    for n in range(n0, max_vertices+1, 2):
        if time.time() > deadline - 4.0:
            break
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            adj = _greedy(n, k, g, rng, deadline-3.0)
            if adj is not None and _girth_ok(adj, n, g):
                return n, _edges(adj, n)
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            adj = _random_k_regular(n, k, rng)
            if adj is not None and _girth_ok(adj, n, g):
                return n, _edges(adj, n)
    return _luw_construct(k, g)
