def construct():
    n = 16
    import random
    random.seed(42)
    
    def is_valid(adj):
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            return False
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    found = False
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            found = True
                            break
                    if not found:
                        return False
        return True
    
    def edge_count(adj):
        return sum(sum(row) for row in adj) // 2
    
    # Initialize with a random sparse graph
    adj = [[False]*n for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.3:
                adj[u][v] = adj[v][u] = True
    
    best_adj = [row[:] for row in adj]
    best_e = edge_count(adj)
    
    # Simulated annealing style local search
    for iteration in range(50000):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        adj[u][v] = not adj[u][v]
        adj[v][u] = adj[u][v]
        
        if is_valid(adj):
            e = edge_count(adj)
            if e > best_e:
                best_e = e
                best_adj = [row[:] for row in adj]
        else:
            adj[u][v] = not adj[u][v]
            adj[v][u] = adj[u][v]
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    return edges
