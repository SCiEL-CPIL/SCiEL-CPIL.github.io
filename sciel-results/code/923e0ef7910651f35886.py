import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    def would_create_c6(adj, u, v):
        nu = adj[u]
        nv = adj[v]
        
        # Calculate N2(u) and N2(v)
        # N2(u) = { w | dist(u,w) == 2 }
        n2_u = set()
        for x in nu:
            n2_u.update(adj[x])
        n2_u.discard(u)
        n2_u.difference_update(nu)
        
        n2_v = set()
        for z in nv:
            n2_v.update(adj[z])
        n2_v.discard(v)
        n2_v.difference_update(nv)
        
        common = n2_u & n2_v
        
        for y in common:
            xs = nu & adj[y]
            zs = nv & adj[y]
            if xs != zs:
                return True
            else:
                if len(xs) > 1:
                    return True
        return False

    def make_k5_graph():
        adj = [set() for _ in range(n)]
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k + a
                    v = 5*k + b
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def make_random_graph(p=0.15):
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def greedy_add(adj):
        changed = True
        while changed:
            changed = False
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u]:
                        candidates.append((u, v))
            random.shuffle(candidates)
            
            for u, v in candidates:
                if v not in adj[u]:
                    if not would_create_c6(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        changed = True
        return adj

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Strategy: Mix of K5 and Random starts
    # K5 starts are dense and structured, likely to find good local optima.
    # Random starts explore different regions.
    
    while time.time() < deadline:
        # 50% K5, 50% Random
        if random.random() < 0.5:
            adj = make_k5_graph()
        else:
            adj = make_random_graph(p=0.1 + random.random()*0.1)
            
        adj = greedy_add(adj)
        edges = get_edges(adj)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    return best_edges
