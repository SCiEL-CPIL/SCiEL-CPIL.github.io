import time
import random
from collections import deque

def _verify_girth_nonbacktracking(adj, g):
    n = len(adj)
    for start in range(n):
        dist = {}
        q = deque()
        for nb in adj[start]:
            dist[(nb, start)] = 1
            q.append((nb, start))
        while q:
            curr, prev = q.popleft()
            d = dist[(curr, prev)]
            if d >= g:
                break
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                new_state = (nxt, curr)
                if new_state in dist:
                    continue
                dist[new_state] = d + 1
                if nxt == start and d + 1 < g:
                    return False
                q.append(new_state)
    return True

def _random_3_regular(n, seed):
    rng = random.Random(seed)
    for _ in range(50):
        stubs = []
        for v in range(n):
            for _ in range(3):
                stubs.append(v)
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n*3, 2):
            u, v = stubs[i], stubs[i+1]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
            else:
                valid = False
                break
        if valid and all(len(adj[v]) == 3 for v in range(n)):
            return [list(s) for s in adj]
    return None

def _luw_construct(k, g):
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
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    if k != 3:
        n = max(g, 2*k)
        edges = [(i, (i+1) % n) for i in range(n)]
        return n, edges
    
    deadline = time.time() + 145
    mb = 2 * sum(2**i for i in range(g//2))
    
    # Try random search
    for n in range(mb, mb + 500):
        if time.time() > deadline:
            break
        for seed in range(30):
            if time.time() > deadline:
                break
            adj = _random_3_regular(n, seed)
            if adj is None:
                continue
            if _verify_girth_nonbacktracking(adj, g):
                edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                return n, edges
    
    # Fallback: LUW
    return _luw_construct(k, g)
