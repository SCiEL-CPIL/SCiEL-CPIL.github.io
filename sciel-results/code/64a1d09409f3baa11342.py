import random
import time
import math

def construct(k, g):
    if k != 4 or g != 10:
        return _fallback(k, g)
    
    deadline = time.time() + 140
    
    # Try to construct a 4-regular graph with girth >= 10
    # Use randomized local search with simulated annealing
    
    def random_4reg(n):
        for _ in range(1000):
            stubs = [v for v in range(n) for _ in range(4)]
            random.shuffle(stubs)
            edges = set()
            ok = True
            for i in range(0, 2*n, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    ok = False
                    break
                key = (min(u,v), max(u,v))
                if key in edges:
                    ok = False
                    break
                edges.add(key)
            if ok:
                return list(edges)
        return None
    
    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def count_4cyc(n, adj):
        c = 0
        for u in range(n):
            for w in range(u+1, n):
                common = adj[u] & adj[w]
                k2 = len(common)
                if k2 >= 2:
                    c += k2*(k2-1)//2
        return c//2
    
    def verify_girth(n, adj, g):
        half = (g-1)//2
        for s in range(n):
            dist = [-1]*n
            par = [-1]*n
            dist[s] = 0
            q = [s]
            for i in range(len(q)):
                u = q[i]
                if dist[u] >= half:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u]+1
                        par[v] = u
                        q.append(v)
                    elif v != par[u] and u != par[v]:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True
    
    # Try different sizes, starting from smaller
    sizes = list(range(244, 322, 2))
    random.shuffle(sizes)
    sizes.sort()  # Try smaller first
    
    for n in sizes:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for init in range(10):
            if time.time() > deadline - 5:
                break
            
            edges = random_4reg(n)
            if edges is None:
                continue
            adj = build_adj(n, edges)
            
            # Simulated annealing
            T = 5.0
            cur_e = count_4cyc(n, adj)
            best_e = cur_e
            best_edges = list(edges)
            
            for step in range(200000):
                if time.time() > deadline - 5:
                    break
                
                # Generate move
                if len(edges) < 2:
                    break
                e1 = random.choice(edges)
                e2 = random.choice(edges)
                if e1 == e2:
                    continue
                u1,v1 = e1
                u2,v2 = e2
                opts = []
                if u1 != u2 and v1 != v2:
                    a = (min(u1,u2), max(u1,u2))
                    b = (min(v1,v2), max(v1,v2))
                    if a != b and a not in edges and b not in edges:
                        opts.append((a,b))
                if u1 != v2 and v1 != u2:
                    a = (min(u1,v2), max(u1,v2))
                    b = (min(v1,u2), max(v1,u2))
                    if a != b and a not in edges and b not in edges:
                        opts.append((a,b))
                if not opts:
                    continue
                a,b = random.choice(opts)
                
                # Apply
                edges.remove(e1)
                edges.remove(e2)
                edges.append(a)
                edges.append(b)
                adj[u1].discard(v1); adj[v1].discard(u1)
                adj[u2].discard(v2); adj[v2].discard(u2)
                adj[a[0]].add(a[1]); adj[a[1]].add(a[0])
                adj[b[0]].add(b[1]); adj[b[1]].add(b[0])
                
                ne = count_4cyc(n, adj)
                delta = ne - cur_e
                if delta < 0 or random.random() < math.exp(-delta / T):
                    cur_e = ne
                    if ne < best_e:
                        best_e = ne
                        best_edges = list(edges)
                        if ne == 0:
                            if verify_girth(n, adj, 10):
                                return n, list(edges)
                else:
                    # Revert
                    edges.remove(a)
                    edges.remove(b)
                    edges.append(e1)
                    edges.append(e2)
                    adj[a[0]].discard(a[1]); adj[a[1]].discard(a[0])
                    adj[b[0]].discard(b[1]); adj[b[1]].discard(b[0])
                    adj[u1].add(v1); adj[v1].add(u1)
                    adj[u2].add(v2); adj[v2].add(u2)
                
                T *= 0.99995
    
    return None

def _fallback(k, g):
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
