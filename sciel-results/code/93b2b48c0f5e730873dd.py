import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def compute_girth_fast(n, edges, check_up_to):
        """Check if girth >= check_up_to using BFS, return actual girth if >= check_up_to"""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            found_short = False
            while queue:
                cur = queue.popleft()
                if dist[cur] >= check_up_to:
                    break
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        parent[nb] = cur
                        if dist[nb] < check_up_to:
                            queue.append(nb)
                    elif parent[nb] != cur and parent[cur] != nb:
                        cycle_len = dist[nb] + dist[cur] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if cycle_len < check_up_to:
                            found_short = True
                            break
                if found_short:
                    break
        return min_girth
    
    # Strategy: Build tensor product C_L x C_M, then use targeted edge switching
    # to eliminate short cycles.
    
    def build_tensor(L, M):
        n = L * M
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(L):
            for j in range(M):
                v = i * M + j
                for di in [-1, 1]:
                    for dj in [-1, 1]:
                        ni = (i + di) % L
                        nj = (j + dj) % M
                        nv = ni * M + nj
                        if v < nv:
                            edges.add((v, nv))
                            adj[v].append(nv)
                            adj[nv].append(v)
        return n, list(edges), adj, edges
    
    # Try C_13 x C_13 = 169 vertices
    best_result = None
    
    for L, M in [(13, 13), (11, 15), (17, 10), (12, 14)]:
        if time.time() > deadline - 10:
            break
        n, edges, adj, edge_set = build_tensor(L, M)
        
        # Count and eliminate short cycles
        for round_num in range(200000):
            if time.time() > deadline - 5:
                break
            
            # Quick check: is girth already >= g?
            girth = compute_girth_fast(n, list(edge_set), g)
            if girth >= g:
                if best_result is None or n < best_result[0]:
                    best_result = (n, list(edge_set))
                break
            
            # Find a short cycle to break
            # Pick random non-adjacent edges for switching
            e1 = random.choice(list(edge_set))
            a, b = e1
            candidates = [e for e in edge_set if e != e1 and e[0] not in (a, b) and e[1] not in (a, b)]
            if not candidates:
                continue
            e2 = random.choice(candidates)
            c, d = e2
            
            switched = False
            for ne1, ne2 in [((a,c),(b,d)), ((a,d),(b,c))]:
                ne1 = (min(ne1[0],ne1[1]), max(ne1[0],ne1[1]))
                ne2 = (min(ne2[0],ne2[1]), max(ne2[0],ne2[1]))
                if ne1[0]==ne1[1] or ne2[0]==ne2[1] or ne1==ne2:
                    continue
                if ne1 in edge_set or ne2 in edge_set:
                    continue
                
                # Apply
                edge_set.discard(e1)
                edge_set.discard(e2)
                adj[a].remove(b); adj[b].remove(a)
                adj[c].remove(d); adj[d].remove(c)
                edge_set.add(ne1); edge_set.add(ne2)
                adj[ne1[0]].append(ne1[1]); adj[ne1[1]].append(ne1[0])
                adj[ne2[0]].append(ne2[1]); adj[ne2[1]].append(ne2[0])
                switched = True
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback
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
    def neighbours(v):
        side, x = v
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for v in order:
        for o in neighbours(v):
            if o not in index:
                index[o] = len(order); order.append(o)
            if v[0]==0:
                edges.append((index[v], index[o]))
    return len(order), edges
