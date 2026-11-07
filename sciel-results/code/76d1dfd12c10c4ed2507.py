import time
import random

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 85
    
    def verify(adj):
        for i in range(n):
            vis = [False]*n
            vis[i] = True
            q = [i]
            md = 0
            d = 1
            while q and d <= 2:
                nq = []
                for u in q:
                    for v in adj[u]:
                        if not vis[v]:
                            vis[v] = True
                            nq.append(v)
                            md = d
                q = nq
                d += 1
            if not all(vis) or md > 2:
                return False
        is_complete = True
        for i in range(n):
            for j in range(i+1, n):
                if j not in adj[i]:
                    is_complete = False
                    break
            if not is_complete:
                break
        if is_complete:
            return False
        
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    ess = False
                    for x in range(n):
                        if x == i or x == j:
                            continue
                        for y in range(x+1, n):
                            if y == i or y == j:
                                continue
                            if y in adj[x]:
                                continue
                            common = adj[x] & adj[y]
                            if not common:
                                continue
                            ok = True
                            for w in common:
                                if not ((x==i and w==j) or (x==j and w==i) or (w==i and y==j) or (w==j and y==i)):
                                    ok = False
                                    break
                            if ok:
                                ess = True
                                break
                        if ess:
                            break
                    if not ess:
                        return False
        return True
    
    best_edges = None
    best_count = 0
    
    def try_candidate(adj, edges):
        nonlocal best_edges, best_count
        if len(edges) <= best_count:
            return
        if verify(adj):
            best_edges = [[i,j] for i,j in edges]
            best_count = len(edges)
    
    # Candidate 1: K(8,8)
    adj1 = [set() for _ in range(n)]
    edges1 = []
    for u in range(8):
        for v in range(8, 16):
            adj1[u].add(v)
            adj1[v].add(u)
            edges1.append((u,v))
    try_candidate(adj1, edges1)
    
    # Candidate 2: Two K8 + perfect matching
    adj2 = [set() for _ in range(n)]
    edges2 = []
    for i in range(8):
        for j in range(i+1, 8):
            adj2[i].add(j)
            adj2[j].add(i)
            edges2.append((i,j))
    for i in range(8, 16):
        for j in range(i+1, 16):
            adj2[i].add(j)
            adj2[j].add(i)
            edges2.append((i,j))
    for i in range(8):
        adj2[i].add(i+8)
        adj2[i+8].add(i)
        edges2.append((i, i+8))
    try_candidate(adj2, edges2)
    
    # Random search
    for trial in range(3000):
        if time.time() > deadline:
            break
        d = 0.15 + random.random() * 0.35
        adj_r = [set() for _ in range(n)]
        edges_r = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < d:
                    adj_r[i].add(j)
                    adj_r[j].add(i)
                    edges_r.append((i,j))
        try_candidate(adj_r, edges_r)
    
    if best_edges is None:
        return []
    return best_edges
